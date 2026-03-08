package relay

import (
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/model"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/service"
	"github.com/QuantumNous/new-api/types"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
)

// ──────────────────────────────────────────────────────────────
// Claude MCP 模式：提取图片 URL 拼接到 user 文本末尾
// ──────────────────────────────────────────────────────────────

func applyClaudeMediaToURL(c *gin.Context, info *relaycommon.RelayInfo, request *dto.ClaudeRequest) *types.NewAPIError {
	if request == nil || len(request.Messages) == 0 {
		return nil
	}

	storedURLBySHA := make(map[string]string)
	imageMaxBytes := int64(constant.MaxImageUploadMB) * 1024 * 1024
	imagePoolMaxBytes := int64(constant.StoredImagePoolMB) * 1024 * 1024

	for i := range request.Messages {
		if strings.ToLower(request.Messages[i].Role) != "user" {
			continue
		}

		contents, err := request.Messages[i].ParseContent()
		if err != nil || len(contents) == 0 {
			continue
		}

		var textParts []string
		var mediaURLs []claudeMediaURL

		for _, part := range contents {
			switch part.Type {
			case dto.ContentTypeText:
				if t := part.GetText(); t != "" {
					textParts = append(textParts, t)
				}
			case "image":
				resolved, rErr := resolveClaudeImageSource(c, info, part.Source, storedURLBySHA, imageMaxBytes, imagePoolMaxBytes)
				if rErr != nil {
					return rErr
				}
				if resolved != "" {
					mediaURLs = append(mediaURLs, claudeMediaURL{Kind: "image", URL: resolved})
				}
			}
		}

		if len(mediaURLs) == 0 {
			continue
		}

		mediaURLs = dedupClaudeMediaURLs(mediaURLs)
		if len(mediaURLs) == 0 {
			continue
		}

		base := strings.TrimRight(strings.Join(textParts, "\n"), " \t\r\n")
		if strings.TrimSpace(base) == "" {
			base = "[media]"
		}

		var b strings.Builder
		b.WriteString(base)
		b.WriteString("\n\n")
		for idx, item := range mediaURLs {
			if idx > 0 {
				b.WriteString("\n")
			}
			b.WriteString("图片URL：")
			b.WriteString(item.URL)
			b.WriteString("，请使用MCP工具查看")
		}

		request.Messages[i].SetStringContent(strings.TrimRight(b.String(), "\n"))
	}

	return nil
}

// ──────────────────────────────────────────────────────────────
// Claude 第三方模型模式：调用第三方多模态模型描述图片内容
// ──────────────────────────────────────────────────────────────

func applyClaudeThirdPartyModelMediaToText(c *gin.Context, info *relaycommon.RelayInfo, request *dto.ClaudeRequest) *types.NewAPIError {
	if request == nil || len(request.Messages) == 0 {
		return nil
	}

	cfg, cfgErr := loadThirdPartyMultimodalConfig()
	if cfgErr != nil {
		return cfgErr
	}

	usingGroup, groupErr := resolveThirdPartyUsingGroup(c, info)
	if groupErr != nil {
		return groupErr
	}

	selected, err := model.GetRandomSatisfiedChannelByAPIType(model.RandomSatisfiedChannelByAPITypeParams{
		Group:     usingGroup,
		ModelName: cfg.modelID,
		APIType:   cfg.callAPIType,
		Retry:     0,
	})
	if err != nil {
		return types.NewError(err, types.ErrorCodeGetChannelFailed, types.ErrOptionWithSkipRetry())
	}
	if selected == nil {
		return types.NewErrorWithStatusCode(
			fmt.Errorf("no available channel for third-party multimodal model %q in group %q", cfg.modelID, usingGroup),
			types.ErrorCodeGetChannelFailed,
			http.StatusBadRequest,
			types.ErrOptionWithSkipRetry(),
		)
	}

	client, buildErr := newThirdPartyMediaTextClient(c, selected, cfg)
	if buildErr != nil {
		return buildErr
	}

	storedURLBySHA := make(map[string]string)
	imageMaxBytes := int64(constant.MaxImageUploadMB) * 1024 * 1024
	imagePoolMaxBytes := int64(constant.StoredImagePoolMB) * 1024 * 1024

	type claudeMediaCounters struct {
		image int
	}
	counters := claudeMediaCounters{}

	for i := range request.Messages {
		if strings.ToLower(request.Messages[i].Role) != "user" {
			continue
		}

		contents, parseErr := request.Messages[i].ParseContent()
		if parseErr != nil || len(contents) == 0 {
			continue
		}

		var textParts []string
		var mediaItems []claudeMediaURL

		for _, part := range contents {
			switch part.Type {
			case dto.ContentTypeText:
				if t := part.GetText(); t != "" {
					textParts = append(textParts, t)
				}
			case "image":
				resolved, rErr := resolveClaudeImageSource(c, info, part.Source, storedURLBySHA, imageMaxBytes, imagePoolMaxBytes)
				if rErr != nil {
					return rErr
				}
				if resolved != "" {
					mediaItems = append(mediaItems, claudeMediaURL{Kind: "image", URL: resolved})
				}
			}
		}

		if len(mediaItems) == 0 {
			continue
		}

		mediaItems = dedupClaudeMediaURLs(mediaItems)
		if len(mediaItems) == 0 {
			continue
		}

		baseText := strings.Join(textParts, "\n")
		if strings.TrimSpace(baseText) == "" {
			baseText = "[media]"
		}

		// Call third-party model to describe each image
		var b strings.Builder
		for _, item := range mediaItems {
			text, descErr := client.Describe(item.Kind, item.URL)
			if descErr != nil {
				return types.NewError(descErr, types.ErrorCodeInvalidRequest, types.ErrOptionWithSkipRetry())
			}
			text = strings.TrimSpace(text)
			if text == "" {
				return types.NewError(
					fmt.Errorf("empty third-party model output for %s: %s", item.Kind, item.URL),
					types.ErrorCodeInvalidRequest,
					types.ErrOptionWithSkipRetry(),
				)
			}

			if b.Len() > 0 {
				b.WriteString("\n")
			}
			counters.image++
			b.WriteString(fmt.Sprintf("图片%d：%s", counters.image, text))
		}

		appendix := strings.TrimRight(b.String(), "\n")
		if strings.TrimSpace(appendix) == "" {
			continue
		}

		newContent := baseText + "\n\n" + appendix
		request.Messages[i].SetStringContent(strings.TrimRight(newContent, "\n"))
	}

	return nil
}

// ──────────────────────────────────────────────────────────────
// Helper: Claude image source → stored URL
// ──────────────────────────────────────────────────────────────

func resolveClaudeImageSource(
	c *gin.Context,
	info *relaycommon.RelayInfo,
	source *dto.ClaudeMessageSource,
	storedURLBySHA map[string]string,
	imageMaxBytes int64,
	imagePoolMaxBytes int64,
) (string, *types.NewAPIError) {
	if source == nil {
		return "", nil
	}

	var rawURL string
	switch source.Type {
	case "url":
		rawURL = strings.TrimSpace(source.Url)
	case "base64":
		dataStr := common.Interface2String(source.Data)
		if dataStr == "" {
			return "", nil
		}
		mediaType := source.MediaType
		if mediaType == "" {
			mediaType = "image/png"
		}
		rawURL = "data:" + mediaType + ";base64," + dataStr
	default:
		return "", nil
	}

	if rawURL == "" {
		return "", nil
	}

	// HTTP URLs can be used directly
	if strings.HasPrefix(rawURL, "http://") || strings.HasPrefix(rawURL, "https://") {
		return rawURL, nil
	}

	// data URL → decode, store, return accessible URL
	mimeType, b64, err := service.DecodeBase64FileData(rawURL)
	if err != nil {
		return "", types.NewErrorWithStatusCode(
			fmt.Errorf("decode Claude image data failed: %w", err),
			types.ErrorCodeInvalidRequest, http.StatusBadRequest, types.ErrOptionWithSkipRetry(),
		)
	}
	mimeType = strings.TrimSpace(mimeType)
	if mimeType == "" {
		return "", types.NewErrorWithStatusCode(
			fmt.Errorf("invalid Claude image mime type: %q", mimeType),
			types.ErrorCodeInvalidRequest, http.StatusBadRequest, types.ErrOptionWithSkipRetry(),
		)
	}
	lowerMime := strings.ToLower(mimeType)
	if !strings.HasPrefix(lowerMime, "image/") {
		return "", types.NewErrorWithStatusCode(
			fmt.Errorf("unsupported Claude media type: %q", mimeType),
			types.ErrorCodeInvalidRequest, http.StatusBadRequest, types.ErrOptionWithSkipRetry(),
		)
	}

	b64 = strings.TrimSpace(b64)
	data, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return "", types.NewErrorWithStatusCode(
			fmt.Errorf("decode Claude image base64 failed: %w", err),
			types.ErrorCodeInvalidRequest, http.StatusBadRequest, types.ErrOptionWithSkipRetry(),
		)
	}
	if len(data) == 0 {
		return "", types.NewErrorWithStatusCode(
			fmt.Errorf("Claude image data is empty"),
			types.ErrorCodeInvalidRequest, http.StatusBadRequest, types.ErrOptionWithSkipRetry(),
		)
	}
	if imageMaxBytes > 0 && int64(len(data)) > imageMaxBytes {
		return "", types.NewErrorWithStatusCode(
			fmt.Errorf("Claude image size %d exceeds limit %d bytes", len(data), imageMaxBytes),
			types.ErrorCodeInvalidRequest, http.StatusBadRequest, types.ErrOptionWithSkipRetry(),
		)
	}

	sha := hex.EncodeToString(common.Sha256Raw(data))
	cacheKey := "image:" + sha
	if existing, ok := storedURLBySHA[cacheKey]; ok {
		return existing, nil
	}

	// Check if already stored in DB
	if existing, err := model.GetStoredImageByUserAndSha(c.Request.Context(), info.UserId, sha); err == nil && existing != nil && existing.Id != "" {
		u := buildStoredImageURL(c, existing.Id)
		storedURLBySHA[cacheKey] = u
		return u, nil
	} else if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
		return "", types.NewError(fmt.Errorf("query stored image failed: %w", err), types.ErrorCodeQueryDataError, types.ErrOptionWithSkipRetry())
	}

	// Store new image
	img := &model.StoredImage{
		UserId:    info.UserId,
		ChannelId: info.ChannelId,
		MimeType:  mimeType,
		SizeBytes: len(data),
		Sha256:    sha,
		Data:      model.LargeBlob(data),
	}
	if err := img.Insert(c.Request.Context()); err != nil {
		return "", types.NewError(fmt.Errorf("store image failed: %w", err), types.ErrorCodeUpdateDataError, types.ErrOptionWithSkipRetry())
	}
	if _, err := model.EnsureStoredImagesPoolLimit(c.Request.Context(), imagePoolMaxBytes, 100); err != nil {
		return "", types.NewError(fmt.Errorf("enforce stored image pool limit failed: %w", err), types.ErrorCodeUpdateDataError, types.ErrOptionWithSkipRetry())
	}

	u := buildStoredImageURL(c, img.Id)
	storedURLBySHA[cacheKey] = u
	return u, nil
}

// ──────────────────────────────────────────────────────────────
// Helper types
// ──────────────────────────────────────────────────────────────

type claudeMediaURL struct {
	Kind string
	URL  string
}

func dedupClaudeMediaURLs(items []claudeMediaURL) []claudeMediaURL {
	if len(items) == 0 {
		return nil
	}
	out := make([]claudeMediaURL, 0, len(items))
	seen := make(map[string]struct{}, len(items))
	for _, item := range items {
		key := item.Kind + "\n" + item.URL
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, item)
	}
	return out
}

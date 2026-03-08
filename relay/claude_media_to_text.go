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
	"github.com/QuantumNous/new-api/logger"
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
		logger.LogInfo(c, "[claude-multimodal-mcp] request is nil or has no messages, skipping")
		return nil
	}

	logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] processing %d messages", len(request.Messages)))

	storedURLBySHA := make(map[string]string)
	imageMaxBytes := int64(constant.MaxImageUploadMB) * 1024 * 1024
	imagePoolMaxBytes := int64(constant.StoredImagePoolMB) * 1024 * 1024

	convertedCount := 0
	for i := range request.Messages {
		role := strings.ToLower(request.Messages[i].Role)
		if role != "user" {
			continue
		}

		contents, err := request.Messages[i].ParseContent()
		if err != nil {
			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d: ParseContent error: %v, checking if string content", i, err))
			// 可能是纯字符串内容
			if request.Messages[i].IsStringContent() {
				logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d: is string content, skipping", i))
			}
			continue
		}
		if len(contents) == 0 {
			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d: no content parts, skipping", i))
			continue
		}

		logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d: %d content parts found", i, len(contents)))

		var textParts []string
		var mediaURLs []claudeMediaURL

		for j, part := range contents {
			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d part %d: type=%q", i, j, part.Type))
			switch part.Type {
			case dto.ContentTypeText:
				if t := part.GetText(); t != "" {
					textParts = append(textParts, t)
					logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d part %d: text length=%d", i, j, len(t)))
				}
			case "image":
				logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d part %d: found image, source=%v", i, j, part.Source != nil))
				resolved, rErr := resolveClaudeImageSource(c, info, part.Source, storedURLBySHA, imageMaxBytes, imagePoolMaxBytes)
				if rErr != nil {
					logger.LogError(c, fmt.Sprintf("[claude-multimodal-mcp] message %d part %d: resolveClaudeImageSource failed: %v", i, j, rErr))
					return rErr
				}
				if resolved != "" {
					logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d part %d: image resolved to URL length=%d", i, j, len(resolved)))
					mediaURLs = append(mediaURLs, claudeMediaURL{Kind: "image", URL: resolved})
				} else {
					logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d part %d: image resolved to empty URL, skipping", i, j))
				}
			default:
				logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d part %d: unhandled type=%q, skipping", i, j, part.Type))
			}
		}

		if len(mediaURLs) == 0 {
			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d: no media URLs found, skipping", i))
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

		newContent := strings.TrimRight(b.String(), "\n")
		request.Messages[i].SetStringContent(newContent)
		convertedCount++
		logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] message %d: converted to text, %d media URLs, new content length=%d", i, len(mediaURLs), len(newContent)))
	}

	logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-mcp] done, converted %d messages total", convertedCount))
	return nil
}

// ──────────────────────────────────────────────────────────────
// Claude 第三方模型模式：调用第三方多模态模型描述图片内容
// ──────────────────────────────────────────────────────────────

func applyClaudeThirdPartyModelMediaToText(c *gin.Context, info *relaycommon.RelayInfo, request *dto.ClaudeRequest) *types.NewAPIError {
	if request == nil || len(request.Messages) == 0 {
		logger.LogInfo(c, "[claude-multimodal-3rd] request is nil or has no messages, skipping")
		return nil
	}

	logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] processing %d messages", len(request.Messages)))

	cfg, cfgErr := loadThirdPartyMultimodalConfig()
	if cfgErr != nil {
		logger.LogError(c, fmt.Sprintf("[claude-multimodal-3rd] loadThirdPartyMultimodalConfig failed: %v", cfgErr))
		return cfgErr
	}
	logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] config loaded: modelID=%s, callAPIType=%d", cfg.modelID, cfg.callAPIType))

	usingGroup, groupErr := resolveThirdPartyUsingGroup(c, info)
	if groupErr != nil {
		logger.LogError(c, fmt.Sprintf("[claude-multimodal-3rd] resolveThirdPartyUsingGroup failed: %v", groupErr))
		return groupErr
	}
	logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] usingGroup=%s", usingGroup))

	selected, err := model.GetRandomSatisfiedChannelByAPIType(model.RandomSatisfiedChannelByAPITypeParams{
		Group:     usingGroup,
		ModelName: cfg.modelID,
		APIType:   cfg.callAPIType,
		Retry:     0,
	})
	if err != nil {
		logger.LogError(c, fmt.Sprintf("[claude-multimodal-3rd] GetRandomSatisfiedChannelByAPIType failed: %v", err))
		return types.NewError(err, types.ErrorCodeGetChannelFailed, types.ErrOptionWithSkipRetry())
	}
	if selected == nil {
		logger.LogError(c, fmt.Sprintf("[claude-multimodal-3rd] no available channel for model=%q group=%q", cfg.modelID, usingGroup))
		return types.NewErrorWithStatusCode(
			fmt.Errorf("no available channel for third-party multimodal model %q in group %q", cfg.modelID, usingGroup),
			types.ErrorCodeGetChannelFailed,
			http.StatusBadRequest,
			types.ErrOptionWithSkipRetry(),
		)
	}
	logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] selected channel: id=%d, type=%d", selected.Id, selected.Type))

	client, buildErr := newThirdPartyMediaTextClient(c, selected, cfg)
	if buildErr != nil {
		logger.LogError(c, fmt.Sprintf("[claude-multimodal-3rd] newThirdPartyMediaTextClient failed: %v", buildErr))
		return buildErr
	}

	storedURLBySHA := make(map[string]string)
	imageMaxBytes := int64(constant.MaxImageUploadMB) * 1024 * 1024
	imagePoolMaxBytes := int64(constant.StoredImagePoolMB) * 1024 * 1024

	type claudeMediaCounters struct {
		image int
	}
	counters := claudeMediaCounters{}
	convertedCount := 0

	for i := range request.Messages {
		role := strings.ToLower(request.Messages[i].Role)
		if role != "user" {
			continue
		}

		contents, parseErr := request.Messages[i].ParseContent()
		if parseErr != nil {
			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d: ParseContent error: %v, checking if string content", i, parseErr))
			if request.Messages[i].IsStringContent() {
				logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d: is string content, skipping", i))
			}
			continue
		}
		if len(contents) == 0 {
			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d: no content parts, skipping", i))
			continue
		}

		logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d: %d content parts found", i, len(contents)))

		var textParts []string
		var mediaItems []claudeMediaURL

		for j, part := range contents {
			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d part %d: type=%q", i, j, part.Type))
			switch part.Type {
			case dto.ContentTypeText:
				if t := part.GetText(); t != "" {
					textParts = append(textParts, t)
				}
			case "image":
				logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d part %d: found image, source=%v", i, j, part.Source != nil))
				if part.Source != nil {
					logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d part %d: source.Type=%s, source.MediaType=%s, source.Url=%q, source.Data_len=%d",
						i, j, part.Source.Type, part.Source.MediaType, part.Source.Url, len(common.Interface2String(part.Source.Data))))
				}
				resolved, rErr := resolveClaudeImageSource(c, info, part.Source, storedURLBySHA, imageMaxBytes, imagePoolMaxBytes)
				if rErr != nil {
					logger.LogError(c, fmt.Sprintf("[claude-multimodal-3rd] message %d part %d: resolveClaudeImageSource failed: %v", i, j, rErr))
					return rErr
				}
				if resolved != "" {
					logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d part %d: image resolved to URL length=%d", i, j, len(resolved)))
					mediaItems = append(mediaItems, claudeMediaURL{Kind: "image", URL: resolved})
				} else {
					logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d part %d: image resolved to empty URL, skipping", i, j))
				}
			default:
				logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d part %d: unhandled type=%q", i, j, part.Type))
			}
		}

		if len(mediaItems) == 0 {
			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d: no media items found, skipping", i))
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

		logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d: calling third-party model for %d images", i, len(mediaItems)))

		// Call third-party model to describe each image
		var b strings.Builder
		for k, item := range mediaItems {
			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d image %d: calling Describe(kind=%s, url_len=%d)", i, k, item.Kind, len(item.URL)))
			text, descErr := client.Describe(item.Kind, item.URL)
			if descErr != nil {
				logger.LogError(c, fmt.Sprintf("[claude-multimodal-3rd] message %d image %d: Describe failed: %v", i, k, descErr))
				return types.NewError(descErr, types.ErrorCodeInvalidRequest, types.ErrOptionWithSkipRetry())
			}
			text = strings.TrimSpace(text)
			if text == "" {
				logger.LogError(c, fmt.Sprintf("[claude-multimodal-3rd] message %d image %d: Describe returned empty text", i, k))
				return types.NewError(
					fmt.Errorf("empty third-party model output for %s: %s", item.Kind, item.URL),
					types.ErrorCodeInvalidRequest,
					types.ErrOptionWithSkipRetry(),
				)
			}

			logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d image %d: Describe returned text length=%d", i, k, len(text)))

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
		convertedCount++
		logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] message %d: converted to text, new content length=%d", i, len(newContent)))
	}

	logger.LogInfo(c, fmt.Sprintf("[claude-multimodal-3rd] done, converted %d messages total", convertedCount))
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
		logger.LogInfo(c, "[claude-resolve-image] source is nil")
		return "", nil
	}

	logger.LogInfo(c, fmt.Sprintf("[claude-resolve-image] source.Type=%s, source.MediaType=%s, source.Url=%q",
		source.Type, source.MediaType, source.Url))

	var rawURL string
	switch source.Type {
	case "url":
		rawURL = strings.TrimSpace(source.Url)
		logger.LogInfo(c, fmt.Sprintf("[claude-resolve-image] url type, rawURL length=%d", len(rawURL)))
	case "base64":
		dataStr := common.Interface2String(source.Data)
		if dataStr == "" {
			logger.LogInfo(c, "[claude-resolve-image] base64 type but data is empty")
			return "", nil
		}
		mediaType := source.MediaType
		if mediaType == "" {
			mediaType = "image/png"
		}
		rawURL = "data:" + mediaType + ";base64," + dataStr
		logger.LogInfo(c, fmt.Sprintf("[claude-resolve-image] base64 type, data length=%d, mediaType=%s, rawURL length=%d", len(dataStr), mediaType, len(rawURL)))
	default:
		logger.LogInfo(c, fmt.Sprintf("[claude-resolve-image] unknown source type: %q", source.Type))
		return "", nil
	}

	if rawURL == "" {
		return "", nil
	}

	// HTTP URLs can be used directly
	if strings.HasPrefix(rawURL, "http://") || strings.HasPrefix(rawURL, "https://") {
		logger.LogInfo(c, fmt.Sprintf("[claude-resolve-image] HTTP URL, using directly, length=%d", len(rawURL)))
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
		logger.LogInfo(c, fmt.Sprintf("[claude-resolve-image] found in local cache, sha=%s", sha[:12]))
		return existing, nil
	}

	// Check if already stored in DB
	if existing, err := model.GetStoredImageByUserAndSha(c.Request.Context(), info.UserId, sha); err == nil && existing != nil && existing.Id != "" {
		u := buildStoredImageURL(c, existing.Id)
		storedURLBySHA[cacheKey] = u
		logger.LogInfo(c, fmt.Sprintf("[claude-resolve-image] found in DB, sha=%s, url=%s", sha[:12], u))
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
	logger.LogInfo(c, fmt.Sprintf("[claude-resolve-image] stored new image, sha=%s, size=%d, url=%s", sha[:12], len(data), u))
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

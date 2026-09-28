package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

type ChatRequest struct {
	Model               string          `json:"model"`
	Messages            []ChatMessage   `json:"messages"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
	Stop                json.RawMessage `json:"stop,omitempty"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	N                   *int            `json:"n,omitempty"`
	User                string          `json:"user,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type ChatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

type ContentPart struct {
	Type     string     `json:"type"`
	Text     string     `json:"text,omitempty"`
	ImageURL *ImageURLP `json:"image_url,omitempty"`
}

type ImageURLP struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function *FunctionDef `json:"function,omitempty"`
}

type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

type ToolChoice struct {
	Type     string `json:"type,omitempty"`
	Function *struct {
		Name string `json:"name"`
	} `json:"function,omitempty"`
}

type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage"`
}

type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type ResponseMessage struct {
	Role             string     `json:"role"`
	Content          *string    `json:"content"`
	ReasoningContent *string    `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

type Usage struct {
	PromptTokens        int                  `json:"prompt_tokens"`
	CompletionTokens    int                  `json:"completion_tokens"`
	TotalTokens         int                  `json:"total_tokens"`
	PromptTokensDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`
}

type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type StreamChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
}

type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type Delta struct {
	Role             string          `json:"role,omitempty"`
	Content          *string         `json:"content,omitempty"`
	ReasoningContent *string         `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCallDelta `json:"tool_calls,omitempty"`
}

type ToolCallDelta struct {
	Index    int            `json:"index"`
	ID       string         `json:"id,omitempty"`
	Type     string         `json:"type,omitempty"`
	Function *FunctionDelta `json:"function,omitempty"`
}

type FunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type openAIError struct {
	Error errorBody `json:"error"`
}

type errorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   any    `json:"param"`
	Code    any    `json:"code"`
}

func newChunk(id string, created int64, model string, choice ChunkChoice) StreamChunk {
	return StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []ChunkChoice{choice},
	}
}

func newChunkWithUsage(id string, created int64, model string, usage *Usage) StreamChunk {
	return StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []ChunkChoice{},
		Usage:   usage,
	}
}

type converseParams struct {
	modelID    string
	system     []types.SystemContentBlock
	messages   []types.Message
	inference  *types.InferenceConfiguration
	toolConfig *types.ToolConfiguration
}

func (p *converseParams) toConverseInput() *bedrockruntime.ConverseInput {
	return &bedrockruntime.ConverseInput{
		ModelId:         aws.String(p.modelID),
		System:          p.system,
		Messages:        p.messages,
		InferenceConfig: p.inference,
		ToolConfig:      p.toolConfig,
	}
}

func (p *converseParams) toStreamInput() *bedrockruntime.ConverseStreamInput {
	return &bedrockruntime.ConverseStreamInput{
		ModelId:         aws.String(p.modelID),
		System:          p.system,
		Messages:        p.messages,
		InferenceConfig: p.inference,
		ToolConfig:      p.toolConfig,
	}
}

func buildParams(req *ChatRequest) (*converseParams, error) {
	p := &converseParams{modelID: resolveModel(req.Model)}

	var raw []types.Message

	for i, m := range req.Messages {
		role := m.Role
		if role == "developer" {
			role = "system"
		}

		switch role {
		case "system":
			text, err := extractText(m.Content)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			if text != "" {
				p.system = append(p.system, &types.SystemContentBlockMemberText{Value: text})
			}

		case "user":
			blocks, err := buildUserBlocks(m.Content)
			if err != nil {
				return nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			if len(blocks) > 0 {
				raw = append(raw, types.Message{Role: types.ConversationRoleUser, Content: blocks})
			}

		case "assistant":
			blocks := buildAssistantBlocks(m, i)
			if len(blocks) > 0 {
				raw = append(raw, types.Message{Role: types.ConversationRoleAssistant, Content: blocks})
			}

		case "tool":
			if m.ToolCallID == "" {
				return nil, fmt.Errorf("messages[%d]: tool message missing tool_call_id", i)
			}
			blocks := buildToolResultBlocks(m)
			raw = append(raw, types.Message{Role: types.ConversationRoleUser, Content: blocks})

		default:
			return nil, fmt.Errorf("messages[%d]: unsupported role %q", i, m.Role)
		}
	}

	p.messages = mergeMessages(raw)

	maxTokens := req.MaxTokens
	if maxTokens == nil {
		maxTokens = req.MaxCompletionTokens
	}
	stopSeqs := parseStop(req.Stop)

	if req.Temperature != nil || req.TopP != nil || maxTokens != nil || len(stopSeqs) > 0 {
		ic := &types.InferenceConfiguration{}
		if req.Temperature != nil {
			t := float32(*req.Temperature)
			if t < 0 {
				t = 0
			}
			if t > 1 {
				t = 1
			}
			ic.Temperature = &t
		}
		if req.TopP != nil {
			tp := float32(*req.TopP)
			ic.TopP = &tp
		}
		if maxTokens != nil && *maxTokens > 0 {
			mt := int32(*maxTokens)
			ic.MaxTokens = &mt
		}
		if len(stopSeqs) > 0 {
			ic.StopSequences = stopSeqs
		}
		p.inference = ic
	}

	if len(req.Tools) > 0 {
		choice, omit, err := parseToolChoice(req.ToolChoice)
		if err != nil {
			return nil, err
		}
		if !omit {
			var tools []types.Tool
			for i, t := range req.Tools {
				if t.Function == nil || t.Function.Name == "" {
					return nil, fmt.Errorf("tools[%d]: function.name is required", i)
				}
				var schema any = map[string]any{"type": "object"}
				if len(t.Function.Parameters) > 0 {
					if err := json.Unmarshal(t.Function.Parameters, &schema); err != nil {
						return nil, fmt.Errorf("tools[%d]: invalid parameters schema: %w", i, err)
					}
				}
				if m, ok := schema.(map[string]any); ok {
					if _, has := m["type"]; !has {
						m["type"] = "object"
					}
				}
				spec := types.ToolSpecification{
					Name:        aws.String(t.Function.Name),
					Description: aws.String(t.Function.Description),
					InputSchema: &types.ToolInputSchemaMemberJson{
						Value: document.NewLazyDocument(schema),
					},
				}
				if t.Function.Strict != nil {
					spec.Strict = t.Function.Strict
				}
				tools = append(tools, &types.ToolMemberToolSpec{Value: spec})
			}
			p.toolConfig = &types.ToolConfiguration{Tools: tools, ToolChoice: choice}
		}
	}

	return p, nil
}

func parseToolChoice(raw json.RawMessage) (types.ToolChoice, bool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, false, nil
	}

	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto", "":
			return &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}, false, nil
		case "required", "any":
			return &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}, false, nil
		case "none":
			return nil, true, nil
		default:
			return nil, false, fmt.Errorf("tool_choice: unsupported value %q", s)
		}
	}

	var tc ToolChoice
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil, false, fmt.Errorf("tool_choice: invalid value")
	}
	if tc.Function != nil && tc.Function.Name != "" {
		return &types.ToolChoiceMemberTool{
			Value: types.SpecificToolChoice{Name: aws.String(tc.Function.Name)},
		}, false, nil
	}
	return &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}, false, nil
}

func parseStop(raw json.RawMessage) []string {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		if one != "" {
			return []string{one}
		}
		return nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		return many
	}
	return nil
}

func extractText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var parts []ContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("content must be string or array of parts")
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String(), nil
}

func buildUserBlocks(raw json.RawMessage) ([]types.ContentBlock, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return nil, nil
		}
		return []types.ContentBlock{&types.ContentBlockMemberText{Value: s}}, nil
	}

	var parts []ContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("content must be string or array of parts")
	}

	var blocks []types.ContentBlock
	for i, p := range parts {
		switch p.Type {
		case "text":
			if p.Text != "" {
				blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Text})
			}
		case "image_url":
			if p.ImageURL == nil || p.ImageURL.URL == "" {
				return nil, fmt.Errorf("content[%d]: image_url.url is required", i)
			}
			img, err := fetchImage(p.ImageURL.URL)
			if err != nil {
				return nil, fmt.Errorf("content[%d]: %w", i, err)
			}
			blocks = append(blocks, &types.ContentBlockMemberImage{Value: *img})
		default:
			return nil, fmt.Errorf("content[%d]: unsupported part type %q", i, p.Type)
		}
	}
	return blocks, nil
}

func buildAssistantBlocks(m ChatMessage, idx int) []types.ContentBlock {
	var blocks []types.ContentBlock

	text, err := extractText(m.Content)
	if err == nil && text != "" {
		blocks = append(blocks, &types.ContentBlockMemberText{Value: text})
	}

	for j, tc := range m.ToolCalls {
		var input any = map[string]any{}
		if tc.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
				input = map[string]any{}
			}
		}
		id := tc.ID
		if id == "" {
			id = fmt.Sprintf("call_msg%d_%d", idx, j)
		}
		name := tc.Function.Name
		blocks = append(blocks, &types.ContentBlockMemberToolUse{
			Value: types.ToolUseBlock{
				ToolUseId: aws.String(id),
				Name:      aws.String(name),
				Input:     document.NewLazyDocument(input),
			},
		})
	}
	return blocks
}

func buildToolResultBlocks(m ChatMessage) []types.ContentBlock {
	text, _ := extractText(m.Content)

	var result types.ToolResultContentBlock
	var parsed any
	if text != "" && json.Unmarshal([]byte(text), &parsed) == nil {
		switch parsed.(type) {
		case map[string]any, []any:
			result = &types.ToolResultContentBlockMemberJson{
				Value: document.NewLazyDocument(parsed),
			}
		default:
			result = &types.ToolResultContentBlockMemberText{Value: text}
		}
	} else {
		result = &types.ToolResultContentBlockMemberText{Value: text}
	}

	return []types.ContentBlock{
		&types.ContentBlockMemberToolResult{
			Value: types.ToolResultBlock{
				ToolUseId: aws.String(m.ToolCallID),
				Content:   []types.ToolResultContentBlock{result},
				Status:    types.ToolResultStatusSuccess,
			},
		},
	}
}

func mergeMessages(msgs []types.Message) []types.Message {
	if len(msgs) == 0 {
		return msgs
	}
	out := []types.Message{msgs[0]}
	for _, m := range msgs[1:] {
		last := &out[len(out)-1]
		if last.Role == m.Role {
			last.Content = append(last.Content, m.Content...)
			continue
		}
		out = append(out, m)
	}
	return out
}

var imageClient = &http.Client{Timeout: 20 * time.Second}

const maxImageBytes = 4 << 20

func fetchImage(url string) (*types.ImageBlock, error) {
	var data []byte
	var mime string

	if strings.HasPrefix(url, "data:") {
		comma := strings.Index(url, ",")
		if comma < 0 {
			return nil, fmt.Errorf("invalid data URI")
		}
		meta := url[5:comma]
		payload := url[comma+1:]
		if !strings.Contains(meta, "base64") {
			return nil, fmt.Errorf("only base64 data URIs supported")
		}
		mime = strings.TrimSuffix(strings.Split(meta, ";")[0], ";")
		var err error
		data, err = base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return nil, fmt.Errorf("invalid base64 image data: %w", err)
		}
	} else if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		resp, err := imageClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("fetch image: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("fetch image: status %d", resp.StatusCode)
		}
		mime = strings.TrimSpace(strings.Split(resp.Header.Get("Content-Type"), ";")[0])
		data, err = io.ReadAll(io.LimitReader(resp.Body, maxImageBytes+1))
		if err != nil {
			return nil, fmt.Errorf("read image: %w", err)
		}
	} else {
		return nil, fmt.Errorf("unsupported image URL scheme (use data: or http(s):)")
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("empty image data")
	}
	if len(data) > maxImageBytes {
		return nil, fmt.Errorf("image exceeds %d bytes", maxImageBytes)
	}

	format := detectImageFormat(data, mime)
	if format == "" {
		return nil, fmt.Errorf("unsupported image format (png, jpeg, gif, webp only)")
	}

	return &types.ImageBlock{
		Format: format,
		Source: &types.ImageSourceMemberBytes{Value: data},
	}, nil
}

func detectImageFormat(data []byte, mime string) types.ImageFormat {
	switch {
	case len(data) >= 8 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4e && data[3] == 0x47:
		return types.ImageFormatPng
	case len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff:
		return types.ImageFormatJpeg
	case len(data) >= 6 && (string(data[:6]) == "GIF87a" || string(data[:6]) == "GIF89a"):
		return types.ImageFormatGif
	case len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP":
		return types.ImageFormatWebp
	}

	switch strings.ToLower(mime) {
	case "image/png":
		return types.ImageFormatPng
	case "image/jpeg", "image/jpg":
		return types.ImageFormatJpeg
	case "image/gif":
		return types.ImageFormatGif
	case "image/webp":
		return types.ImageFormatWebp
	}
	return ""
}

package bedrock

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"lm-prox/internal/openai"
)

type Params struct {
	modelID    string
	system     []types.SystemContentBlock
	messages   []types.Message
	inference  *types.InferenceConfiguration
	toolConfig *types.ToolConfiguration
}

func (p *Params) toConverseInput() *bedrockruntime.ConverseInput {
	return &bedrockruntime.ConverseInput{
		ModelId:         aws.String(p.modelID),
		System:          p.system,
		Messages:        p.messages,
		InferenceConfig: p.inference,
		ToolConfig:      p.toolConfig,
	}
}

func (p *Params) toStreamInput() *bedrockruntime.ConverseStreamInput {
	return &bedrockruntime.ConverseStreamInput{
		ModelId:         aws.String(p.modelID),
		System:          p.system,
		Messages:        p.messages,
		InferenceConfig: p.inference,
		ToolConfig:      p.toolConfig,
	}
}

func BuildParams(req *openai.ChatRequest) (*Params, error) {
	p := &Params{modelID: Resolve(req.Model)}

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

	var tc openai.ToolChoice
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
	var parts []openai.ContentPart
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

	var parts []openai.ContentPart
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

func buildAssistantBlocks(m openai.ChatMessage, idx int) []types.ContentBlock {
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

func buildToolResultBlocks(m openai.ChatMessage) []types.ContentBlock {
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

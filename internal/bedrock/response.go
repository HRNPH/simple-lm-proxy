package bedrock

import (
	"encoding/json"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"lm-prox/internal/openai"
)

func BuildCompletion(req *openai.ChatRequest, out *bedrockruntime.ConverseOutput) *openai.ChatCompletionResponse {
	msg, _ := out.Output.(*types.ConverseOutputMemberMessage)

	var content string
	var reasoning string
	var toolCalls []openai.ToolCall

	if msg != nil {
		for _, block := range msg.Value.Content {
			switch b := block.(type) {
			case *types.ContentBlockMemberText:
				content += b.Value
			case *types.ContentBlockMemberReasoningContent:
				if rt, ok := b.Value.(*types.ReasoningContentBlockMemberReasoningText); ok {
					if reasoning != "" {
						reasoning += "\n\n"
					}
					reasoning += aws.ToString(rt.Value.Text)
				}
			case *types.ContentBlockMemberToolUse:
				var input any
				if err := b.Value.Input.UnmarshalSmithyDocument(&input); err != nil {
					input = map[string]any{}
				}
				args, _ := json.Marshal(input)
				toolCalls = append(toolCalls, openai.ToolCall{
					ID:   aws.ToString(b.Value.ToolUseId),
					Type: "function",
					Function: openai.FunctionCall{
						Name:      aws.ToString(b.Value.Name),
						Arguments: string(args),
					},
				})
			}
		}
	}

	fr := mapStopReason(out.StopReason)
	if len(toolCalls) > 0 && fr == "stop" {
		fr = "tool_calls"
	}

	msgOut := openai.ResponseMessage{Role: "assistant"}
	if content != "" || len(toolCalls) == 0 {
		msgOut.Content = openai.StrPtr(content)
	}
	if reasoning != "" {
		msgOut.ReasoningContent = openai.StrPtr(reasoning)
	}
	if len(toolCalls) > 0 {
		msgOut.ToolCalls = toolCalls
	}

	return &openai.ChatCompletionResponse{
		ID:      openai.NewID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []openai.Choice{{
			Index:        0,
			Message:      msgOut,
			FinishReason: fr,
		}},
		Usage: mapUsage(out.Usage),
	}
}

func mapUsage(u *types.TokenUsage) *openai.Usage {
	if u == nil {
		return &openai.Usage{}
	}
	usage := &openai.Usage{
		PromptTokens:     int(aws.ToInt32(u.InputTokens)),
		CompletionTokens: int(aws.ToInt32(u.OutputTokens)),
		TotalTokens:      int(aws.ToInt32(u.TotalTokens)),
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	if cached := int(aws.ToInt32(u.CacheReadInputTokens)); cached > 0 {
		usage.PromptTokensDetails = &openai.PromptTokensDetails{CachedTokens: cached}
	}
	return usage
}

func mapStopReason(s types.StopReason) string {
	switch s {
	case types.StopReasonToolUse:
		return "tool_calls"
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return "length"
	case types.StopReasonContentFiltered, types.StopReasonGuardrailIntervened:
		return "content_filter"
	default:
		return "stop"
	}
}

package bedrock

import (
	"context"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"lm-prox/internal/openai"
)

type StreamMeta struct {
	ID      string
	Created int64
	Model   string
}

func (c *Client) Stream(ctx context.Context, p *Params, meta StreamMeta, emit func(any)) error {
	out, err := c.inner.ConverseStream(ctx, p.toStreamInput())
	if err != nil {
		return err
	}

	stream := out.GetStream()
	defer stream.Close()

	emit(openai.NewChunk(meta.ID, meta.Created, meta.Model, openai.ChunkChoice{
		Index: 0,
		Delta: openai.Delta{Role: "assistant", Content: openai.StrPtr("")},
	}))

	toolIdxByBlock := map[int32]int{}
	nextToolIdx := 0
	var usage *openai.Usage
	finished := false

	for event := range stream.Events() {
		switch e := event.(type) {
		case *types.ConverseStreamOutputMemberContentBlockStart:
			if tu, ok := e.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
				idx := nextToolIdx
				nextToolIdx++
				toolIdxByBlock[aws.ToInt32(e.Value.ContentBlockIndex)] = idx
				emit(openai.NewChunk(meta.ID, meta.Created, meta.Model, openai.ChunkChoice{
					Index: 0,
					Delta: openai.Delta{
						ToolCalls: []openai.ToolCallDelta{{
							Index:    idx,
							ID:       aws.ToString(tu.Value.ToolUseId),
							Type:     "function",
							Function: &openai.FunctionDelta{Name: aws.ToString(tu.Value.Name)},
						}},
					},
				}))
			}

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			switch dt := e.Value.Delta.(type) {
			case *types.ContentBlockDeltaMemberText:
				emit(openai.NewChunk(meta.ID, meta.Created, meta.Model, openai.ChunkChoice{
					Index: 0,
					Delta: openai.Delta{Content: openai.StrPtr(dt.Value)},
				}))
			case *types.ContentBlockDeltaMemberReasoningContent:
				if rt, ok := dt.Value.(*types.ReasoningContentBlockDeltaMemberText); ok {
					emit(openai.NewChunk(meta.ID, meta.Created, meta.Model, openai.ChunkChoice{
						Index: 0,
						Delta: openai.Delta{ReasoningContent: openai.StrPtr(rt.Value)},
					}))
				}
			case *types.ContentBlockDeltaMemberToolUse:
				idx, ok := toolIdxByBlock[aws.ToInt32(e.Value.ContentBlockIndex)]
				if !ok {
					continue
				}
				args := aws.ToString(dt.Value.Input)
				if args == "" {
					continue
				}
				emit(openai.NewChunk(meta.ID, meta.Created, meta.Model, openai.ChunkChoice{
					Index: 0,
					Delta: openai.Delta{
						ToolCalls: []openai.ToolCallDelta{{
							Index:    idx,
							Function: &openai.FunctionDelta{Arguments: args},
						}},
					},
				}))
			}

		case *types.ConverseStreamOutputMemberMessageStop:
			finished = true
			fr := mapStopReason(e.Value.StopReason)
			if fr == "stop" && nextToolIdx > 0 {
				fr = "tool_calls"
			}
			emit(openai.NewChunk(meta.ID, meta.Created, meta.Model, openai.ChunkChoice{
				Index:        0,
				Delta:        openai.Delta{},
				FinishReason: &fr,
			}))

		case *types.ConverseStreamOutputMemberMetadata:
			usage = mapUsage(e.Value.Usage)
		}
	}

	if serr := stream.Err(); serr != nil {
		log.Printf("stream error: %v", serr)
		if !finished {
			emit(openai.OpenAIError{
				Error: openai.ErrorBody{
					Message: serr.Error(),
					Type:    "api_error",
				},
			})
		}
	}

	if !finished {
		fr := "stop"
		if nextToolIdx > 0 {
			fr = "tool_calls"
		}
		emit(openai.NewChunk(meta.ID, meta.Created, meta.Model, openai.ChunkChoice{
			Index:        0,
			Delta:        openai.Delta{},
			FinishReason: &fr,
		}))
	}

	if usage != nil {
		emit(openai.NewChunkWithUsage(meta.ID, meta.Created, meta.Model, usage))
	}

	return nil
}

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go"
)

const (
	defaultPort   = "10000"
	defaultRegion = "us-east-1"
)

var (
	brClient *bedrockruntime.Client
	region   string
)

var modelAliases = map[string]string{
	"opus-4.6":          "global.anthropic.claude-opus-4-6-v1",
	"claude-opus-4.6":   "global.anthropic.claude-opus-4-6-v1",
	"opus-4.5":          "global.anthropic.claude-opus-4-5-20251101-v1:0",
	"claude-opus-4.5":   "global.anthropic.claude-opus-4-5-20251101-v1:0",
	"opus-4.7":          "global.anthropic.claude-opus-4-7",
	"opus-4.8":          "global.anthropic.claude-opus-4-8",
	"opus-5":            "global.anthropic.claude-opus-5",
	"opus-5.5":          "global.anthropic.claude-opus-5-5",
	"sonnet-4.6":        "global.anthropic.claude-sonnet-4-6",
	"claude-sonnet-4.6": "global.anthropic.claude-sonnet-4-6",
	"sonnet-4.5":        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
	"sonnet-5":          "global.anthropic.claude-sonnet-5",
	"haiku-4.5":         "global.anthropic.claude-haiku-4-5-20251001-v1:0",
	"kimi-k3":           "global.moonshotai.kimi-k3",
	"moonshot-kimi-k3":  "global.moonshotai.kimi-k3",
	"grok-4.6":          "global.xai.grok-4.6",
	"gpt-5.4":           "global.openai.gpt-5.4",
	"gpt-5.5":           "global.openai.gpt-5.5",
	"gpt-5.6-sol":       "global.openai.gpt-5.6-sol",
	"gpt-5.6-luna":      "global.openai.gpt-5.6-luna",
	"gpt-5.6-terra":     "global.openai.gpt-5.6-terra",
	"gpt-6-astra":       "global.openai.gpt-6-astra",
	"gpt-6-sol":         "global.openai.gpt-6-sol",
	"gpt-6-luna":        "global.openai.gpt-6-luna",
	"nova-pro":          "global.amazon.nova-pro-v1:0",
	"nova-lite":         "global.amazon.nova-lite-v1:0",
	"nova-micro":        "global.amazon.nova-micro-v1:0",
}

func resolveModel(m string) string {
	if v, ok := modelAliases[strings.ToLower(strings.TrimSpace(m))]; ok {
		return v
	}
	return m
}

func main() {
	region = os.Getenv("AWS_REGION")
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		region = defaultRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatalf("load aws config: %v", err)
	}
	brClient = bedrockruntime.NewFromConfig(cfg)

	authMode := "aws-credential-chain"
	if os.Getenv("AWS_BEARER_TOKEN_BEDROCK") != "" {
		authMode = "bedrock-bearer-token"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("GET /v1/models", handleModels)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"ok":true}`)
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = defaultPort
	}

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           withCORS(withLogging(mux)),
		ReadHeaderTimeout: 15 * time.Second,
	}

	go func() {
		log.Printf("lm-prox listening on :%s region=%s auth=%s", port, region, authMode)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	log.Println("shutting down")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(sw, r)
		log.Printf("%s %s -> %d (%s)", r.Method, r.URL.Path, sw.status, time.Since(start).Round(time.Millisecond))
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	seen := map[string]bool{}
	var ids []string
	for alias := range modelAliases {
		ids = append(ids, alias)
	}
	for _, v := range modelAliases {
		if !seen[v] {
			seen[v] = true
			ids = append(ids, v)
		}
	}
	sort.Strings(ids)

	data := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		data = append(data, map[string]any{
			"id":       id,
			"object":   "model",
			"created":  0,
			"owned_by": "amazon-bedrock",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 32<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON body: "+err.Error())
		return
	}
	if strings.TrimSpace(req.Model) == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "messages is required")
		return
	}

	params, err := buildParams(&req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if req.Stream {
		streamChat(w, r, &req, params)
		return
	}

	out, err := brClient.Converse(r.Context(), params.toConverseInput())
	if err != nil {
		st, msg, typ := mapError(err)
		writeError(w, st, typ, msg)
		return
	}
	resp := buildCompletion(&req, out)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func streamChat(w http.ResponseWriter, r *http.Request, req *ChatRequest, params *converseParams) {
	out, err := brClient.ConverseStream(r.Context(), params.toStreamInput())
	if err != nil {
		st, msg, typ := mapError(err)
		writeError(w, st, typ, msg)
		return
	}

	rc := http.NewResponseController(w)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	id := newID()
	created := time.Now().Unix()
	model := req.Model

	emit := func(v any) {
		b, mErr := json.Marshal(v)
		if mErr != nil {
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", b)
		_ = rc.Flush()
	}

	emit(newChunk(id, created, model, ChunkChoice{
		Index: 0,
		Delta: Delta{Role: "assistant", Content: strPtr("")},
	}))

	stream := out.GetStream()
	defer stream.Close()

	toolIdxByBlock := map[int32]int{}
	nextToolIdx := 0
	var usage *Usage
	finished := false

	for event := range stream.Events() {
		switch e := event.(type) {
		case *types.ConverseStreamOutputMemberContentBlockStart:
			if tu, ok := e.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
				idx := nextToolIdx
				nextToolIdx++
				toolIdxByBlock[aws.ToInt32(e.Value.ContentBlockIndex)] = idx
				emit(newChunk(id, created, model, ChunkChoice{
					Index: 0,
					Delta: Delta{
						ToolCalls: []ToolCallDelta{{
							Index:    idx,
							ID:       aws.ToString(tu.Value.ToolUseId),
							Type:     "function",
							Function: &FunctionDelta{Name: aws.ToString(tu.Value.Name)},
						}},
					},
				}))
			}

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			switch dt := e.Value.Delta.(type) {
			case *types.ContentBlockDeltaMemberText:
				emit(newChunk(id, created, model, ChunkChoice{
					Index: 0,
					Delta: Delta{Content: strPtr(dt.Value)},
				}))
			case *types.ContentBlockDeltaMemberReasoningContent:
				if rt, ok := dt.Value.(*types.ReasoningContentBlockDeltaMemberText); ok {
					emit(newChunk(id, created, model, ChunkChoice{
						Index: 0,
						Delta: Delta{ReasoningContent: strPtr(rt.Value)},
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
				emit(newChunk(id, created, model, ChunkChoice{
					Index: 0,
					Delta: Delta{
						ToolCalls: []ToolCallDelta{{
							Index:    idx,
							Function: &FunctionDelta{Arguments: args},
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
			emit(newChunk(id, created, model, ChunkChoice{
				Index:        0,
				Delta:        Delta{},
				FinishReason: &fr,
			}))

		case *types.ConverseStreamOutputMemberMetadata:
			usage = mapUsage(e.Value.Usage)
		}
	}

	if serr := stream.Err(); serr != nil {
		log.Printf("stream error: %v", serr)
		if !finished {
			b, _ := json.Marshal(openAIError{
				Error: errorBody{
					Message: serr.Error(),
					Type:    "api_error",
				},
			})
			fmt.Fprintf(w, "data: %s\n\n", b)
		}
	}

	if !finished {
		fr := "stop"
		if nextToolIdx > 0 {
			fr = "tool_calls"
		}
		emit(newChunk(id, created, model, ChunkChoice{
			Index:        0,
			Delta:        Delta{},
			FinishReason: &fr,
		}))
	}

	if usage != nil {
		emit(newChunkWithUsage(id, created, model, usage))
	}

	fmt.Fprint(w, "data: [DONE]\n\n")
	_ = rc.Flush()
}

func buildCompletion(req *ChatRequest, out *bedrockruntime.ConverseOutput) *ChatCompletionResponse {
	msg, _ := out.Output.(*types.ConverseOutputMemberMessage)

	var content string
	var reasoning string
	var toolCalls []ToolCall

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
				toolCalls = append(toolCalls, ToolCall{
					ID:   aws.ToString(b.Value.ToolUseId),
					Type: "function",
					Function: FunctionCall{
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

	msgOut := ResponseMessage{Role: "assistant"}
	if content != "" || len(toolCalls) == 0 {
		msgOut.Content = strPtr(content)
	}
	if reasoning != "" {
		msgOut.ReasoningContent = strPtr(reasoning)
	}
	if len(toolCalls) > 0 {
		msgOut.ToolCalls = toolCalls
	}

	return &ChatCompletionResponse{
		ID:      newID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []Choice{{
			Index:        0,
			Message:      msgOut,
			FinishReason: fr,
		}},
		Usage: mapUsage(out.Usage),
	}
}

func mapUsage(u *types.TokenUsage) *Usage {
	if u == nil {
		return &Usage{}
	}
	usage := &Usage{
		PromptTokens:     int(aws.ToInt32(u.InputTokens)),
		CompletionTokens: int(aws.ToInt32(u.OutputTokens)),
		TotalTokens:      int(aws.ToInt32(u.TotalTokens)),
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	if cached := int(aws.ToInt32(u.CacheReadInputTokens)); cached > 0 {
		usage.PromptTokensDetails = &PromptTokensDetails{CachedTokens: cached}
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

func mapError(err error) (int, string, string) {
	var ae smithy.APIError
	if errors.As(err, &ae) {
		msg := ae.ErrorMessage()
		switch ae.ErrorCode() {
		case "ValidationException":
			return http.StatusBadRequest, msg, "invalid_request_error"
		case "AccessDeniedException", "AccessDenied":
			return http.StatusForbidden, msg, "permission_error"
		case "UnauthorizedException", "UnrecognizedClientException", "InvalidSignatureException":
			return http.StatusUnauthorized, msg, "authentication_error"
		case "ThrottlingException", "Throttling", "TooManyRequestsException", "ServiceQuotaExceededException":
			return http.StatusTooManyRequests, msg, "rate_limit_error"
		case "ResourceNotFoundException":
			return http.StatusNotFound, msg, "invalid_request_error"
		case "ModelNotReadyException":
			return http.StatusServiceUnavailable, msg, "server_error"
		case "ServiceUnavailableException", "InternalServerException":
			return http.StatusInternalServerError, msg, "server_error"
		case "ModelTimeoutException":
			return http.StatusGatewayTimeout, msg, "server_error"
		default:
			return http.StatusInternalServerError, ae.ErrorCode() + ": " + msg, "api_error"
		}
	}

	msg := err.Error()
	lower := strings.ToLower(msg)
	if strings.Contains(lower, "sso session") ||
		strings.Contains(lower, "sso login") ||
		(strings.Contains(lower, "credential") && strings.Contains(lower, "expired")) ||
		strings.Contains(lower, "failed to get shared config profile") ||
		strings.Contains(lower, "expired or is otherwise invalid") {
		profile := os.Getenv("AWS_PROFILE")
		hint := "run: aws sso login"
		if profile != "" {
			hint = fmt.Sprintf("run: aws sso login --profile %s", profile)
		}
		return http.StatusUnauthorized,
			fmt.Sprintf("AWS credentials unavailable: %s (or set AWS_BEARER_TOKEN_BEDROCK)", hint),
			"authentication_error"
	}
	if strings.Contains(lower, "failed to retrieve") && strings.Contains(lower, "token") {
		return http.StatusUnauthorized,
			"AWS bearer token invalid - regenerate Bedrock API key and set AWS_BEARER_TOKEN_BEDROCK",
			"authentication_error"
	}
	return http.StatusInternalServerError, msg, "api_error"
}

func writeError(w http.ResponseWriter, status int, errType, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(openAIError{
		Error: errorBody{
			Message: msg,
			Type:    errType,
		},
	})
}

func newID() string {
	var b [12]byte
	rand.Read(b[:])
	return "chatcmpl-" + hex.EncodeToString(b[:])
}

func strPtr(s string) *string { return &s }

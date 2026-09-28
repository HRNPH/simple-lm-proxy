package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"lm-prox/internal/bedrock"
	"lm-prox/internal/openai"
)

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	seen := map[string]bool{}
	var ids []string
	for alias := range bedrock.Aliases {
		ids = append(ids, alias)
	}
	for _, v := range bedrock.Aliases {
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

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req openai.ChatRequest
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

	params, err := bedrock.BuildParams(&req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if req.Stream {
		s.streamChat(w, r, &req, params)
		return
	}

	out, err := s.client.Converse(r.Context(), params)
	if err != nil {
		st, msg, typ := mapError(err)
		writeError(w, st, typ, msg)
		return
	}
	resp := bedrock.BuildCompletion(&req, out)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) streamChat(w http.ResponseWriter, r *http.Request, req *openai.ChatRequest, params *bedrock.Params) {
	rc := http.NewResponseController(w)
	started := false

	emit := func(v any) {
		if !started {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("X-Accel-Buffering", "no")
			w.WriteHeader(http.StatusOK)
			started = true
		}
		b, err := json.Marshal(v)
		if err != nil {
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", b)
		_ = rc.Flush()
	}

	meta := bedrock.StreamMeta{
		ID:      openai.NewID(),
		Created: time.Now().Unix(),
		Model:   req.Model,
	}
	if err := s.client.Stream(r.Context(), params, meta, emit); err != nil {
		st, msg, typ := mapError(err)
		writeError(w, st, typ, msg)
		return
	}

	if !started {
		emit(openai.NewChunk(meta.ID, meta.Created, meta.Model, openai.ChunkChoice{
			Index: 0,
			Delta: openai.Delta{Role: "assistant", Content: openai.StrPtr("")},
		}))
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	_ = rc.Flush()
}

func writeError(w http.ResponseWriter, status int, errType, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(openai.OpenAIError{
		Error: openai.ErrorBody{
			Message: msg,
			Type:    errType,
		},
	})
}

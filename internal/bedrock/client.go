package bedrock

import (
	"context"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const DefaultRegion = "us-east-1"

var Aliases = map[string]string{
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

func Resolve(m string) string {
	if v, ok := Aliases[strings.ToLower(strings.TrimSpace(m))]; ok {
		return v
	}
	return m
}

type Client struct {
	inner *bedrockruntime.Client
}

func New(region string) (*Client, error) {
	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		return nil, fmt.Errorf("load aws config: %w", err)
	}
	return &Client{inner: bedrockruntime.NewFromConfig(cfg)}, nil
}

func (c *Client) Converse(ctx context.Context, p *Params) (*bedrockruntime.ConverseOutput, error) {
	return c.inner.Converse(ctx, p.toConverseInput())
}

package apicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestChatCompletionsToCompletions_BasicText(t *testing.T) {
	maxTokens := 100
	maxCompletionTokens := 42
	temperature := 0.7
	topP := 0.9
	req := &ChatCompletionsRequest{
		Model:               "gpt-legacy",
		Instructions:        "Follow policy.",
		Messages:            []ChatMessage{{Role: "system", Content: json.RawMessage(`"You are helpful."`)}, {Role: "user", Content: json.RawMessage(`"Hello"`)}, {Role: "assistant", Content: json.RawMessage(`"Hi"`)}},
		MaxTokens:           &maxTokens,
		MaxCompletionTokens: &maxCompletionTokens,
		Temperature:         &temperature,
		TopP:                &topP,
		Stream:              true,
		Stop:                json.RawMessage(`["\nUser:"]`),
	}

	got, err := ChatCompletionsToCompletions(req)
	require.NoError(t, err)
	require.Equal(t, "gpt-legacy", got.Model)
	require.Equal(t, "System: Follow policy.\nSystem: You are helpful.\nUser: Hello\nAssistant: Hi\nAssistant:", got.Prompt)
	require.NotNil(t, got.MaxTokens)
	require.Equal(t, 42, *got.MaxTokens)
	require.Equal(t, temperature, *got.Temperature)
	require.Equal(t, topP, *got.TopP)
	require.True(t, got.Stream)
	require.JSONEq(t, `["\nUser:"]`, string(got.Stop))
}

func TestChatCompletionsToCompletions_TextParts(t *testing.T) {
	req := &ChatCompletionsRequest{
		Model: "gpt-legacy",
		Messages: []ChatMessage{{
			Role:    "user",
			Content: json.RawMessage(`[{"type":"text","text":"hello "},{"type":"text","text":"world"}]`),
		}},
	}

	got, err := ChatCompletionsToCompletions(req)
	require.NoError(t, err)
	require.Equal(t, "User: hello world\nAssistant:", got.Prompt)
}

func TestChatCompletionsToCompletions_RejectsUnsupportedFeatures(t *testing.T) {
	tests := []struct {
		name string
		req  *ChatCompletionsRequest
	}{
		{"tools", &ChatCompletionsRequest{Model: "m", Tools: []ChatTool{{Type: "function"}}}},
		{"functions", &ChatCompletionsRequest{Model: "m", Functions: []ChatFunction{{Name: "f"}}}},
		{"tool choice", &ChatCompletionsRequest{Model: "m", ToolChoice: json.RawMessage(`"auto"`)}},
		{"function call", &ChatCompletionsRequest{Model: "m", FunctionCall: json.RawMessage(`"auto"`)}},
		{"tool calls", &ChatCompletionsRequest{Model: "m", Messages: []ChatMessage{{Role: "assistant", ToolCalls: []ChatToolCall{{ID: "call_1"}}}}}},
		{"tool message", &ChatCompletionsRequest{Model: "m", Messages: []ChatMessage{{Role: "tool", Content: json.RawMessage(`"x"`)}}}},
		{"image content", &ChatCompletionsRequest{Model: "m", Messages: []ChatMessage{{Role: "user", Content: json.RawMessage(`[{"type":"image_url","image_url":{"url":"data:image/png;base64,xxx"}}]`)}}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ChatCompletionsToCompletions(tt.req)
			require.Error(t, err)
		})
	}
}

func TestCompletionsToChatCompletions(t *testing.T) {
	resp := &CompletionsResponse{
		ID:      "cmpl_1",
		Created: 123,
		Model:   "upstream-model",
		Choices: []CompletionsChoice{{Index: 0, Text: "hello", FinishReason: "length"}},
		Usage:   &CompletionsUsage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10, PromptTokensDetails: &ChatTokenDetails{CachedTokens: 2}},
	}

	got := CompletionsToChatCompletions(resp, "client-model")
	require.Equal(t, "cmpl_1", got.ID)
	require.Equal(t, "chat.completion", got.Object)
	require.Equal(t, int64(123), got.Created)
	require.Equal(t, "client-model", got.Model)
	require.Len(t, got.Choices, 1)
	require.Equal(t, "assistant", got.Choices[0].Message.Role)
	require.JSONEq(t, `"hello"`, string(got.Choices[0].Message.Content))
	require.Equal(t, "length", got.Choices[0].FinishReason)
	require.Equal(t, 7, got.Usage.PromptTokens)
	require.Equal(t, 3, got.Usage.CompletionTokens)
	require.Equal(t, 10, got.Usage.TotalTokens)
	require.Equal(t, 2, got.Usage.PromptTokensDetails.CachedTokens)
}

func TestCompletionsChunkToChatChunks(t *testing.T) {
	state := NewCompletionsEventToChatState()
	state.Model = "client-model"
	finish := "stop"
	chunk := &CompletionsChunk{
		ID:      "cmpl_1",
		Created: 123,
		Model:   "upstream-model",
		Choices: []CompletionsChunkChoice{{Index: 0, Text: "hello", FinishReason: &finish}},
		Usage:   &CompletionsUsage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6},
	}

	got := CompletionsChunkToChatChunks(chunk, state)
	require.Len(t, got, 3)
	require.Equal(t, "assistant", got[0].Choices[0].Delta.Role)
	require.NotNil(t, got[1].Choices[0].Delta.Content)
	require.Equal(t, "hello", *got[1].Choices[0].Delta.Content)
	require.NotNil(t, got[2].Choices[0].FinishReason)
	require.Equal(t, "stop", *got[2].Choices[0].FinishReason)
	require.Equal(t, 4, state.Usage.PromptTokens)
}

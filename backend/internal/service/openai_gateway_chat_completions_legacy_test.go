//go:build unit

package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestBuildOpenAICompletionsURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		base string
		want string
	}{
		{"already /v1/completions", "https://api.openai.com/v1/completions", "https://api.openai.com/v1/completions"},
		{"already /completions", "https://legacy.example/completions", "https://legacy.example/completions"},
		{"bare /v1", "https://api.openai.com/v1", "https://api.openai.com/v1/completions"},
		{"bare domain", "https://api.openai.com", "https://api.openai.com/v1/completions"},
		{"trailing slash", "https://api.openai.com/", "https://api.openai.com/v1/completions"},
		{"chat completions is not legacy completions", "https://api.openai.com/v1/chat/completions", "https://api.openai.com/v1/chat/completions/v1/completions"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tt.want, buildOpenAICompletionsURL(tt.base))
		})
	}
}

func TestForwardAsLegacyCompletionsFromChat_Buffered(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-legacy","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":12}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "X-Request-Id": []string{"rid_legacy"}},
		Body: io.NopCloser(strings.NewReader(`{
			"id":"cmpl_1",
			"object":"text_completion",
			"created":123,
			"model":"upstream-model",
			"choices":[{"text":"world","index":0,"finish_reason":"stop"}],
			"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,"prompt_tokens_details":{"cached_tokens":1}}
		}`)),
	}}
	account := legacyCompletionsTestAccount()
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream}

	result, err := svc.forwardAsLegacyCompletionsFromChat(context.Background(), c, account, body, "", "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "rid_legacy", result.RequestID)
	require.Equal(t, 5, result.Usage.InputTokens)
	require.Equal(t, 2, result.Usage.OutputTokens)
	require.Equal(t, 1, result.Usage.CacheReadInputTokens)
	require.NotNil(t, upstream.lastReq)
	require.Equal(t, "http://upstream.example/v1/completions", upstream.lastReq.URL.String())
	require.Equal(t, "Bearer sk-test", upstream.lastReq.Header.Get("Authorization"))
	require.Equal(t, "gpt-legacy", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, "User: hello\nAssistant:", gjson.GetBytes(upstream.lastBody, "prompt").String())
	require.Equal(t, int64(12), gjson.GetBytes(upstream.lastBody, "max_tokens").Int())
	require.False(t, gjson.GetBytes(upstream.lastBody, "messages").Exists())
	require.JSONEq(t, `{"id":"cmpl_1","object":"chat.completion","created":123,"model":"gpt-legacy","choices":[{"index":0,"message":{"role":"assistant","content":"world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7,"prompt_tokens_details":{"cached_tokens":1}}}`, rec.Body.String())
}

func TestForwardAsLegacyCompletionsFromChat_Stream(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-legacy","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		`data: {"id":"cmpl_1","object":"text_completion","created":123,"model":"upstream-model","choices":[{"text":"he","index":0,"finish_reason":null}]}`,
		"",
		`data: {"id":"cmpl_1","object":"text_completion","created":123,"model":"upstream-model","choices":[{"text":"llo","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "X-Request-Id": []string{"rid_legacy_stream"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	account := legacyCompletionsTestAccount()
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream}

	result, err := svc.forwardAsLegacyCompletionsFromChat(context.Background(), c, account, body, "", "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.Equal(t, 4, result.Usage.InputTokens)
	require.Equal(t, 2, result.Usage.OutputTokens)
	require.NotNil(t, result.FirstTokenMs)
	require.Equal(t, "http://upstream.example/v1/completions", upstream.lastReq.URL.String())
	require.Equal(t, "text/event-stream", upstream.lastReq.Header.Get("Accept"))
	require.True(t, gjson.GetBytes(upstream.lastBody, "stream").Bool())
	out := rec.Body.String()
	require.Contains(t, out, `"object":"chat.completion.chunk"`)
	require.Contains(t, out, `"role":"assistant"`)
	require.Contains(t, out, `"content":"he"`)
	require.Contains(t, out, `"content":"llo"`)
	require.Contains(t, out, `"finish_reason":"stop"`)
	require.Contains(t, out, `"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}`)
	require.Contains(t, out, "data: [DONE]")
}

func TestForwardAsChatCompletions_LegacyCompletionsMode(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-legacy","messages":[{"role":"user","content":"hello"}]}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(`{"id":"cmpl_1","choices":[{"text":"ok","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)),
	}}
	account := legacyCompletionsTestAccount()
	account.Extra = map[string]any{openai_compat.ExtraKeyUpstreamAPI: string(openai_compat.OpenAIUpstreamAPILegacyCompletions)}
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream}

	_, err := svc.ForwardAsChatCompletions(context.Background(), c, account, body, "", "")
	require.NoError(t, err)
	require.Equal(t, "http://upstream.example/v1/completions", upstream.lastReq.URL.String())
}

func legacyCompletionsTestAccount() *Account {
	account := rawChatCompletionsTestAccount()
	account.Name = "legacy-openai-apikey"
	return account
}

package openai_compat

import "testing"

func TestResolveResponsesSupport(t *testing.T) {
	tests := []struct {
		name  string
		extra map[string]any
		want  AccountResponsesSupport
	}{
		{"nil extra", nil, ResponsesSupportUnknown},
		{"empty extra", map[string]any{}, ResponsesSupportUnknown},
		{"key missing", map[string]any{"other": "value"}, ResponsesSupportUnknown},
		{"value true", map[string]any{ExtraKeyResponsesSupported: true}, ResponsesSupportYes},
		{"value false", map[string]any{ExtraKeyResponsesSupported: false}, ResponsesSupportNo},
		{"value wrong type string", map[string]any{ExtraKeyResponsesSupported: "true"}, ResponsesSupportUnknown},
		{"value wrong type number", map[string]any{ExtraKeyResponsesSupported: 1}, ResponsesSupportUnknown},
		{"value nil", map[string]any{ExtraKeyResponsesSupported: nil}, ResponsesSupportUnknown},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ResolveResponsesSupport(tc.extra)
			if got != tc.want {
				t.Errorf("ResolveResponsesSupport(%v) = %v, want %v", tc.extra, got, tc.want)
			}
		})
	}
}

func TestShouldUseResponsesAPI(t *testing.T) {
	tests := []struct {
		name  string
		extra map[string]any
		want  bool
	}{
		// 关键不变量：未探测必须返回 true（保留旧行为）
		{"unknown defaults to true (preserve old behavior)", nil, true},
		{"unknown empty defaults to true", map[string]any{}, true},
		{"unknown wrong type defaults to true", map[string]any{ExtraKeyResponsesSupported: "yes"}, true},

		// 已探测：标记决定
		{"explicitly supported", map[string]any{ExtraKeyResponsesSupported: true}, true},
		{"explicitly unsupported", map[string]any{ExtraKeyResponsesSupported: false}, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ShouldUseResponsesAPI(tc.extra)
			if got != tc.want {
				t.Errorf("ShouldUseResponsesAPI(%v) = %v, want %v", tc.extra, got, tc.want)
			}
		})
	}
}

func TestResolveChatCompletionsUpstreamAPI(t *testing.T) {
	tests := []struct {
		name  string
		extra map[string]any
		want  OpenAIUpstreamAPI
	}{
		{"nil extra defaults to responses", nil, OpenAIUpstreamAPIResponses},
		{"empty extra defaults to responses", map[string]any{}, OpenAIUpstreamAPIResponses},
		{"responses supported defaults to responses", map[string]any{ExtraKeyResponsesSupported: true}, OpenAIUpstreamAPIResponses},
		{"responses unsupported defaults to chat completions", map[string]any{ExtraKeyResponsesSupported: false}, OpenAIUpstreamAPIChatCompletions},
		{"explicit legacy completions wins over missing probe", map[string]any{ExtraKeyUpstreamAPI: "legacy_completions"}, OpenAIUpstreamAPILegacyCompletions},
		{"explicit legacy completions wins over supported probe", map[string]any{ExtraKeyUpstreamAPI: "legacy_completions", ExtraKeyResponsesSupported: true}, OpenAIUpstreamAPILegacyCompletions},
		{"explicit legacy completions wins over unsupported probe", map[string]any{ExtraKeyUpstreamAPI: "legacy_completions", ExtraKeyResponsesSupported: false}, OpenAIUpstreamAPILegacyCompletions},
		{"explicit chat completions wins over supported probe", map[string]any{ExtraKeyUpstreamAPI: "chat_completions", ExtraKeyResponsesSupported: true}, OpenAIUpstreamAPIChatCompletions},
		{"explicit responses wins over unsupported probe", map[string]any{ExtraKeyUpstreamAPI: "responses", ExtraKeyResponsesSupported: false}, OpenAIUpstreamAPIResponses},
		{"explicit value is trimmed and case insensitive", map[string]any{ExtraKeyUpstreamAPI: " Legacy_Completions "}, OpenAIUpstreamAPILegacyCompletions},
		{"auto follows unsupported probe", map[string]any{ExtraKeyUpstreamAPI: "auto", ExtraKeyResponsesSupported: false}, OpenAIUpstreamAPIChatCompletions},
		{"unknown explicit value follows probe", map[string]any{ExtraKeyUpstreamAPI: "completions", ExtraKeyResponsesSupported: false}, OpenAIUpstreamAPIChatCompletions},
		{"non string explicit value follows probe", map[string]any{ExtraKeyUpstreamAPI: 1, ExtraKeyResponsesSupported: false}, OpenAIUpstreamAPIChatCompletions},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ResolveChatCompletionsUpstreamAPI(tc.extra)
			if got != tc.want {
				t.Errorf("ResolveChatCompletionsUpstreamAPI(%v) = %v, want %v", tc.extra, got, tc.want)
			}
		})
	}
}

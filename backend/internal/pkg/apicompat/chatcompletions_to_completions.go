package apicompat

import (
	"encoding/json"
	"fmt"
	"strings"
)

func ChatCompletionsToCompletions(req *ChatCompletionsRequest) (*CompletionsRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("request is nil")
	}
	if err := validateChatCompletionsRequestForLegacyCompletions(req); err != nil {
		return nil, err
	}
	prompt, err := chatCompletionsPrompt(req)
	if err != nil {
		return nil, err
	}

	out := &CompletionsRequest{
		Model:       req.Model,
		Prompt:      prompt,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Stop:        req.Stop,
	}
	if req.MaxTokens != nil {
		out.MaxTokens = req.MaxTokens
	}
	if req.MaxCompletionTokens != nil {
		out.MaxTokens = req.MaxCompletionTokens
	}
	return out, nil
}

func validateChatCompletionsRequestForLegacyCompletions(req *ChatCompletionsRequest) error {
	if len(req.Tools) > 0 {
		return fmt.Errorf("legacy completions upstream does not support tools")
	}
	if len(req.Functions) > 0 {
		return fmt.Errorf("legacy completions upstream does not support functions")
	}
	if rawJSONIsMeaningful(req.ToolChoice) {
		return fmt.Errorf("legacy completions upstream does not support tool_choice")
	}
	if rawJSONIsMeaningful(req.FunctionCall) {
		return fmt.Errorf("legacy completions upstream does not support function_call")
	}
	for _, msg := range req.Messages {
		if len(msg.ToolCalls) > 0 {
			return fmt.Errorf("legacy completions upstream does not support tool_calls")
		}
		if msg.FunctionCall != nil {
			return fmt.Errorf("legacy completions upstream does not support function_call messages")
		}
		if msg.Role == "tool" || msg.Role == "function" {
			return fmt.Errorf("legacy completions upstream does not support %s messages", msg.Role)
		}
	}
	return nil
}

func chatCompletionsPrompt(req *ChatCompletionsRequest) (string, error) {
	var lines []string
	if strings.TrimSpace(req.Instructions) != "" {
		lines = append(lines, "System: "+req.Instructions)
	}
	for _, msg := range req.Messages {
		text, err := chatMessageContentTextOnly(msg.Content)
		if err != nil {
			return "", fmt.Errorf("convert %s message content: %w", msg.Role, err)
		}
		lines = append(lines, legacyPromptRoleLabel(msg.Role)+": "+text)
	}
	lines = append(lines, "Assistant:")
	return strings.Join(lines, "\n"), nil
}

func legacyPromptRoleLabel(role string) string {
	switch role {
	case "system", "developer":
		return "System"
	case "assistant":
		return "Assistant"
	case "user", "":
		return "User"
	default:
		return strings.ToUpper(role[:1]) + role[1:]
	}
}

func chatMessageContentTextOnly(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var parts []ChatContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("expected string or text content parts")
	}
	var b strings.Builder
	for _, part := range parts {
		switch part.Type {
		case "text":
			_, _ = b.WriteString(part.Text)
		case "":
			continue
		default:
			return "", fmt.Errorf("legacy completions upstream does not support %s content", part.Type)
		}
	}
	return b.String(), nil
}

func rawJSONIsMeaningful(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && trimmed != "null"
}

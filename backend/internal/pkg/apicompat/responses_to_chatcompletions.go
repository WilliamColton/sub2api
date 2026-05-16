package apicompat

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Request conversion: ResponsesRequest → ChatCompletionsRequest
// ---------------------------------------------------------------------------

// ResponsesRequestToChatCompletions converts a Responses API request into a
// Chat Completions request. This is used when the upstream only supports
// /v1/chat/completions (e.g., DeepSeek, Kimi, GLM, Qwen).
func ResponsesRequestToChatCompletions(req *ResponsesRequest) (*ChatCompletionsRequest, error) {
	messages, err := convertResponsesInputToChatMessages(req.Input, req.Instructions)
	if err != nil {
		return nil, fmt.Errorf("convert responses input to chat messages: %w", err)
	}

	out := &ChatCompletionsRequest{
		Model:       req.Model,
		Messages:    messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		ServiceTier: req.ServiceTier,
	}

	if req.MaxOutputTokens != nil {
		out.MaxTokens = req.MaxOutputTokens
	}

	if len(req.Tools) > 0 {
		out.Tools = convertResponsesToolsToChatTools(req.Tools)
	}

	if req.ToolChoice != nil {
		out.ToolChoice = req.ToolChoice
	}

	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		out.ReasoningEffort = req.Reasoning.Effort
	}

	return out, nil
}

func convertResponsesInputToChatMessages(input json.RawMessage, instructions string) ([]ChatMessage, error) {
	var messages []ChatMessage

	// Add instructions as system message
	if instructions != "" {
		messages = append(messages, ChatMessage{
			Role:    "system",
			Content: json.RawMessage(`"` + instructions + `"`),
		})
	}

	// Try to parse as array of ResponsesInputItem
	var items []ResponsesInputItem
	if err := json.Unmarshal(input, &items); err != nil {
		// Try to parse as plain string
		var s string
		if err2 := json.Unmarshal(input, &s); err2 != nil {
			return nil, fmt.Errorf("unmarshal responses input: %w", err)
		}
		if s != "" {
			messages = append(messages, ChatMessage{
				Role:    "user",
				Content: json.RawMessage(`"` + s + `"`),
			})
		}
		return messages, nil
	}

	for _, item := range items {
		switch item.Type {
		case "function_call":
			messages = append(messages, ChatMessage{
				Role:      "assistant",
				ToolCalls: []ChatToolCall{{
					ID:   item.CallID,
					Type: "function",
					Function: ChatFunctionCall{
						Name:      item.Name,
						Arguments: item.Arguments,
					},
				}},
			})
		case "function_call_output":
			messages = append(messages, ChatMessage{
				Role:       "tool",
				Content:    json.RawMessage(`"` + item.Output + `"`),
				ToolCallID: item.CallID,
			})
		default:
			// Role-based messages (developer/system/user/assistant)
			role := item.Role
			if role == "" {
				role = "user"
			}
			if role == "developer" {
				role = "system"
			}

			content := convertResponsesContentToChatContent(item.Content)
			messages = append(messages, ChatMessage{
				Role:    role,
				Content: content,
			})
		}
	}

	return messages, nil
}

// convertResponsesContentToChatContent converts Responses API content format
// to Chat Completions content format. The key difference is:
// - Responses uses "input_text", "output_text", "input_image"
// - Chat Completions uses "text", "image_url"
func convertResponsesContentToChatContent(content json.RawMessage) json.RawMessage {
	if content == nil {
		return json.RawMessage(`""`)
	}

	// Try to parse as string (simple case)
	var s string
	if err := json.Unmarshal(content, &s); err == nil {
		return content
	}

	// Try to parse as array of content parts
	var parts []ResponsesContentPart
	if err := json.Unmarshal(content, &parts); err != nil {
		// Return as-is if we can't parse it
		return content
	}

	// Convert to ChatContentPart format
	var chatParts []ChatContentPart
	for _, part := range parts {
		switch part.Type {
		case "input_text", "output_text":
			chatParts = append(chatParts, ChatContentPart{
				Type: "text",
				Text: part.Text,
			})
		case "input_image":
			chatParts = append(chatParts, ChatContentPart{
				Type: "image_url",
				ImageURL: &ChatImageURL{
					URL: part.ImageURL,
				},
			})
		default:
			// Pass through unknown types as text
			chatParts = append(chatParts, ChatContentPart{
				Type: "text",
				Text: part.Text,
			})
		}
	}

	if len(chatParts) == 0 {
		return json.RawMessage(`""`)
	}

	result, err := json.Marshal(chatParts)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return result
}

func convertResponsesToolsToChatTools(tools []ResponsesTool) []ChatTool {
	var chatTools []ChatTool
	for _, t := range tools {
		if t.Type == "function" {
			chatTools = append(chatTools, ChatTool{
				Type: "function",
				Function: &ChatFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
					Strict:      t.Strict,
				},
			})
		}
		// Skip non-function tools (web_search, local_shell, etc.) as they
		// don't have a direct Chat Completions equivalent
	}
	return chatTools
}

// ---------------------------------------------------------------------------
// Non-streaming: ResponsesResponse → ChatCompletionsResponse
// ---------------------------------------------------------------------------

// ResponsesToChatCompletions converts a Responses API response into a Chat
// Completions response. Text output items are concatenated into
// choices[0].message.content; function_call items become tool_calls.
func ResponsesToChatCompletions(resp *ResponsesResponse, model string) *ChatCompletionsResponse {
	id := resp.ID
	if id == "" {
		id = generateChatCmplID()
	}

	out := &ChatCompletionsResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
	}

	var contentText string
	var reasoningText string
	var toolCalls []ChatToolCall

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" && part.Text != "" {
					contentText += part.Text
				}
			}
		case "function_call":
			toolCalls = append(toolCalls, ChatToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: ChatFunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
		case "reasoning":
			for _, s := range item.Summary {
				if s.Type == "summary_text" && s.Text != "" {
					reasoningText += s.Text
				}
			}
		case "web_search_call":
			// silently consumed — results already incorporated into text output
		}
	}

	msg := ChatMessage{Role: "assistant"}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}
	if contentText != "" {
		raw, _ := json.Marshal(contentText)
		msg.Content = raw
	}
	if reasoningText != "" {
		msg.ReasoningContent = reasoningText
	}

	finishReason := responsesStatusToChatFinishReason(resp.Status, resp.IncompleteDetails, toolCalls)

	out.Choices = []ChatChoice{{
		Index:        0,
		Message:      msg,
		FinishReason: finishReason,
	}}

	if resp.Usage != nil {
		usage := &ChatUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		}
		if resp.Usage.InputTokensDetails != nil && resp.Usage.InputTokensDetails.CachedTokens > 0 {
			usage.PromptTokensDetails = &ChatTokenDetails{
				CachedTokens: resp.Usage.InputTokensDetails.CachedTokens,
			}
		}
		out.Usage = usage
	}

	return out
}

func responsesStatusToChatFinishReason(status string, details *ResponsesIncompleteDetails, toolCalls []ChatToolCall) string {
	switch status {
	case "incomplete":
		if details != nil && details.Reason == "max_output_tokens" {
			return "length"
		}
		return "stop"
	case "completed":
		if len(toolCalls) > 0 {
			return "tool_calls"
		}
		return "stop"
	default:
		return "stop"
	}
}

// ---------------------------------------------------------------------------
// Streaming: ResponsesStreamEvent → []ChatCompletionsChunk (stateful converter)
// ---------------------------------------------------------------------------

// ResponsesEventToChatState tracks state for converting a sequence of Responses
// SSE events into Chat Completions SSE chunks.
type ResponsesEventToChatState struct {
	ID                     string
	Model                  string
	Created                int64
	SentRole               bool
	SawToolCall            bool
	SawText                bool
	Finalized              bool        // true after finish chunk has been emitted
	NextToolCallIndex      int         // next sequential tool_call index to assign
	OutputIndexToToolIndex map[int]int // Responses output_index → Chat tool_calls index
	IncludeUsage           bool
	Usage                  *ChatUsage
}

// NewResponsesEventToChatState returns an initialised stream state.
func NewResponsesEventToChatState() *ResponsesEventToChatState {
	return &ResponsesEventToChatState{
		ID:                     generateChatCmplID(),
		Created:                time.Now().Unix(),
		OutputIndexToToolIndex: make(map[int]int),
	}
}

// ResponsesEventToChatChunks converts a single Responses SSE event into zero
// or more Chat Completions chunks, updating state as it goes.
func ResponsesEventToChatChunks(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	switch evt.Type {
	case "response.created":
		return resToChatHandleCreated(evt, state)
	case "response.output_text.delta":
		return resToChatHandleTextDelta(evt, state)
	case "response.output_item.added":
		return resToChatHandleOutputItemAdded(evt, state)
	case "response.function_call_arguments.delta":
		return resToChatHandleFuncArgsDelta(evt, state)
	case "response.reasoning_summary_text.delta":
		return resToChatHandleReasoningDelta(evt, state)
	case "response.reasoning_summary_text.done":
		return nil
	// response.done 是 Realtime/WS 与项目透传路径使用的终止别名；
	// 普通 Responses HTTP SSE 的公开终止事件仍以 response.completed 为主。
	case "response.completed", "response.done", "response.incomplete", "response.failed":
		return resToChatHandleCompleted(evt, state)
	default:
		return nil
	}
}

// FinalizeResponsesChatStream emits a final chunk with finish_reason if the
// stream ended without a proper completion event (e.g. upstream disconnect).
// It is idempotent: if a completion event already emitted the finish chunk,
// this returns nil.
func FinalizeResponsesChatStream(state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if state.Finalized {
		return nil
	}
	state.Finalized = true

	finishReason := "stop"
	if state.SawToolCall {
		finishReason = "tool_calls"
	}

	chunks := []ChatCompletionsChunk{makeChatFinishChunk(state, finishReason)}

	if state.IncludeUsage && state.Usage != nil {
		chunks = append(chunks, ChatCompletionsChunk{
			ID:      state.ID,
			Object:  "chat.completion.chunk",
			Created: state.Created,
			Model:   state.Model,
			Choices: []ChatChunkChoice{},
			Usage:   state.Usage,
		})
	}

	return chunks
}

// ChatChunkToSSE formats a ChatCompletionsChunk as an SSE data line.
func ChatChunkToSSE(chunk ChatCompletionsChunk) (string, error) {
	data, err := json.Marshal(chunk)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("data: %s\n\n", data), nil
}

// --- internal handlers ---

func resToChatHandleCreated(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Response != nil {
		if evt.Response.ID != "" {
			state.ID = evt.Response.ID
		}
		if state.Model == "" && evt.Response.Model != "" {
			state.Model = evt.Response.Model
		}
	}
	// Emit the role chunk.
	if state.SentRole {
		return nil
	}
	state.SentRole = true

	role := "assistant"
	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{Role: role})}
}

func resToChatHandleTextDelta(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Delta == "" {
		return nil
	}
	state.SawText = true
	content := evt.Delta
	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{Content: &content})}
}

func resToChatHandleOutputItemAdded(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Item == nil || evt.Item.Type != "function_call" {
		return nil
	}

	state.SawToolCall = true
	idx := state.NextToolCallIndex
	state.OutputIndexToToolIndex[evt.OutputIndex] = idx
	state.NextToolCallIndex++

	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{
		ToolCalls: []ChatToolCall{{
			Index: &idx,
			ID:    evt.Item.CallID,
			Type:  "function",
			Function: ChatFunctionCall{
				Name: evt.Item.Name,
			},
		}},
	})}
}

func resToChatHandleFuncArgsDelta(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Delta == "" {
		return nil
	}

	idx, ok := state.OutputIndexToToolIndex[evt.OutputIndex]
	if !ok {
		return nil
	}

	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{
		ToolCalls: []ChatToolCall{{
			Index: &idx,
			Function: ChatFunctionCall{
				Arguments: evt.Delta,
			},
		}},
	})}
}

func resToChatHandleReasoningDelta(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	if evt.Delta == "" {
		return nil
	}
	reasoning := evt.Delta
	return []ChatCompletionsChunk{makeChatDeltaChunk(state, ChatDelta{ReasoningContent: &reasoning})}
}

func resToChatHandleCompleted(evt *ResponsesStreamEvent, state *ResponsesEventToChatState) []ChatCompletionsChunk {
	state.Finalized = true
	finishReason := "stop"

	if evt.Response != nil {
		if evt.Response.Usage != nil {
			u := evt.Response.Usage
			usage := &ChatUsage{
				PromptTokens:     u.InputTokens,
				CompletionTokens: u.OutputTokens,
				TotalTokens:      u.InputTokens + u.OutputTokens,
			}
			if u.InputTokensDetails != nil && u.InputTokensDetails.CachedTokens > 0 {
				usage.PromptTokensDetails = &ChatTokenDetails{
					CachedTokens: u.InputTokensDetails.CachedTokens,
				}
			}
			state.Usage = usage
		}

		switch evt.Response.Status {
		case "incomplete":
			if evt.Response.IncompleteDetails != nil && evt.Response.IncompleteDetails.Reason == "max_output_tokens" {
				finishReason = "length"
			}
		case "completed":
			if state.SawToolCall {
				finishReason = "tool_calls"
			}
		}
	} else if state.SawToolCall {
		finishReason = "tool_calls"
	}

	var chunks []ChatCompletionsChunk
	chunks = append(chunks, makeChatFinishChunk(state, finishReason))

	if state.IncludeUsage && state.Usage != nil {
		chunks = append(chunks, ChatCompletionsChunk{
			ID:      state.ID,
			Object:  "chat.completion.chunk",
			Created: state.Created,
			Model:   state.Model,
			Choices: []ChatChunkChoice{},
			Usage:   state.Usage,
		})
	}

	return chunks
}

func makeChatDeltaChunk(state *ResponsesEventToChatState, delta ChatDelta) ChatCompletionsChunk {
	return ChatCompletionsChunk{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []ChatChunkChoice{{
			Index:        0,
			Delta:        delta,
			FinishReason: nil,
		}},
	}
}

func makeChatFinishChunk(state *ResponsesEventToChatState, finishReason string) ChatCompletionsChunk {
	empty := ""
	return ChatCompletionsChunk{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []ChatChunkChoice{{
			Index:        0,
			Delta:        ChatDelta{Content: &empty},
			FinishReason: &finishReason,
		}},
	}
}

// generateChatCmplID returns a "chatcmpl-" prefixed random hex ID.
func generateChatCmplID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "chatcmpl-" + hex.EncodeToString(b)
}

// ---------------------------------------------------------------------------
// BufferedResponseAccumulator: accumulates SSE delta events for non-streaming
// paths where the terminal event may have empty output.
// ---------------------------------------------------------------------------

type bufferedFuncCall struct {
	CallID string
	Name   string
	Args   strings.Builder
}

// BufferedResponseAccumulator collects content from Responses SSE delta events
// so that non-streaming handlers can reconstruct output when the terminal event
// (response.completed / response.done) carries an empty output array.
type BufferedResponseAccumulator struct {
	text                 strings.Builder
	reasoning            strings.Builder
	funcCalls            []bufferedFuncCall
	outputIndexToFuncIdx map[int]int
}

// NewBufferedResponseAccumulator returns an initialised accumulator.
func NewBufferedResponseAccumulator() *BufferedResponseAccumulator {
	return &BufferedResponseAccumulator{
		outputIndexToFuncIdx: make(map[int]int),
	}
}

// ProcessEvent inspects a single Responses SSE event and accumulates any
// content it carries. Only delta events that contribute to the final output
// are handled; all other event types are silently ignored.
func (a *BufferedResponseAccumulator) ProcessEvent(event *ResponsesStreamEvent) {
	switch event.Type {
	case "response.output_text.delta":
		if event.Delta != "" {
			_, _ = a.text.WriteString(event.Delta)
		}
	case "response.output_item.added":
		if event.Item != nil && event.Item.Type == "function_call" {
			idx := len(a.funcCalls)
			a.outputIndexToFuncIdx[event.OutputIndex] = idx
			a.funcCalls = append(a.funcCalls, bufferedFuncCall{
				CallID: event.Item.CallID,
				Name:   event.Item.Name,
			})
		}
	case "response.function_call_arguments.delta":
		if event.Delta != "" {
			if idx, ok := a.outputIndexToFuncIdx[event.OutputIndex]; ok {
				_, _ = a.funcCalls[idx].Args.WriteString(event.Delta)
			}
		}
	case "response.reasoning_summary_text.delta":
		if event.Delta != "" {
			_, _ = a.reasoning.WriteString(event.Delta)
		}
	}
}

// HasContent reports whether any content has been accumulated.
func (a *BufferedResponseAccumulator) HasContent() bool {
	return a.text.Len() > 0 || len(a.funcCalls) > 0 || a.reasoning.Len() > 0
}

// BuildOutput constructs a []ResponsesOutput from the accumulated delta
// content. The order matches what ResponsesToChatCompletions expects:
// reasoning → message → function_calls.
func (a *BufferedResponseAccumulator) BuildOutput() []ResponsesOutput {
	var out []ResponsesOutput

	if a.reasoning.Len() > 0 {
		out = append(out, ResponsesOutput{
			Type: "reasoning",
			Summary: []ResponsesSummary{{
				Type: "summary_text",
				Text: a.reasoning.String(),
			}},
		})
	}

	if a.text.Len() > 0 {
		out = append(out, ResponsesOutput{
			Type: "message",
			Role: "assistant",
			Content: []ResponsesContentPart{{
				Type: "output_text",
				Text: a.text.String(),
			}},
		})
	}

	for i := range a.funcCalls {
		out = append(out, ResponsesOutput{
			Type:      "function_call",
			CallID:    a.funcCalls[i].CallID,
			Name:      a.funcCalls[i].Name,
			Arguments: a.funcCalls[i].Args.String(),
		})
	}

	return out
}

// SupplementResponseOutput fills resp.Output from accumulated delta content
// when the terminal event delivered an empty output array. If resp.Output is
// already populated, this is a no-op (preserves backward compatibility).
func (a *BufferedResponseAccumulator) SupplementResponseOutput(resp *ResponsesResponse) {
	if resp == nil || len(resp.Output) > 0 {
		return
	}
	if !a.HasContent() {
		return
	}
	resp.Output = a.BuildOutput()
}

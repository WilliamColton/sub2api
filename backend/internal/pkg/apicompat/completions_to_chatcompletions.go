package apicompat

import (
	"encoding/json"
	"time"
)

func CompletionsToChatCompletions(resp *CompletionsResponse, model string) *ChatCompletionsResponse {
	id := ""
	created := time.Now().Unix()
	if resp != nil {
		id = resp.ID
		if resp.Created > 0 {
			created = resp.Created
		}
	}
	if id == "" {
		id = generateChatCmplID()
	}

	out := &ChatCompletionsResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
	}
	if resp == nil {
		return out
	}

	out.Choices = make([]ChatChoice, 0, len(resp.Choices))
	for _, choice := range resp.Choices {
		content, _ := jsonMarshalString(choice.Text)
		out.Choices = append(out.Choices, ChatChoice{
			Index: choice.Index,
			Message: ChatMessage{
				Role:    "assistant",
				Content: content,
			},
			FinishReason: normalizeCompletionFinishReason(choice.FinishReason),
		})
	}
	if len(out.Choices) == 0 {
		content, _ := jsonMarshalString("")
		out.Choices = []ChatChoice{{
			Index: 0,
			Message: ChatMessage{
				Role:    "assistant",
				Content: content,
			},
			FinishReason: "stop",
		}}
	}
	out.Usage = completionsUsageToChatUsage(resp.Usage)
	return out
}

type CompletionsEventToChatState struct {
	ID        string
	Model     string
	Created   int64
	SentRole  map[int]bool
	Finalized map[int]bool
	Usage     *ChatUsage
}

func NewCompletionsEventToChatState() *CompletionsEventToChatState {
	return &CompletionsEventToChatState{
		ID:        generateChatCmplID(),
		Created:   time.Now().Unix(),
		SentRole:  make(map[int]bool),
		Finalized: make(map[int]bool),
	}
}

func CompletionsChunkToChatChunks(chunk *CompletionsChunk, state *CompletionsEventToChatState) []ChatCompletionsChunk {
	if chunk == nil || state == nil {
		return nil
	}
	if chunk.ID != "" {
		state.ID = chunk.ID
	}
	if chunk.Created > 0 {
		state.Created = chunk.Created
	}
	if state.Model == "" && chunk.Model != "" {
		state.Model = chunk.Model
	}
	if chunk.Usage != nil {
		state.Usage = completionsUsageToChatUsage(chunk.Usage)
	}

	var chunks []ChatCompletionsChunk
	for _, choice := range chunk.Choices {
		index := choice.Index
		if !state.SentRole[index] {
			state.SentRole[index] = true
			chunks = append(chunks, makeCompletionChatChunk(state, index, ChatDelta{Role: "assistant"}, nil))
		}
		if choice.Text != "" {
			text := choice.Text
			chunks = append(chunks, makeCompletionChatChunk(state, index, ChatDelta{Content: &text}, nil))
		}
		if choice.FinishReason != nil {
			finishReason := normalizeCompletionFinishReason(*choice.FinishReason)
			state.Finalized[index] = true
			empty := ""
			chunks = append(chunks, makeCompletionChatChunk(state, index, ChatDelta{Content: &empty}, &finishReason))
		}
	}
	return chunks
}

func FinalizeCompletionsChatStream(state *CompletionsEventToChatState) []ChatCompletionsChunk {
	if state == nil || state.Finalized[0] {
		return nil
	}
	finishReason := "stop"
	empty := ""
	var chunks []ChatCompletionsChunk
	if !state.SentRole[0] {
		state.SentRole[0] = true
		chunks = append(chunks, makeCompletionChatChunk(state, 0, ChatDelta{Role: "assistant"}, nil))
	}
	chunks = append(chunks, makeCompletionChatChunk(state, 0, ChatDelta{Content: &empty}, &finishReason))
	return chunks
}

func makeCompletionChatChunk(state *CompletionsEventToChatState, index int, delta ChatDelta, finishReason *string) ChatCompletionsChunk {
	return ChatCompletionsChunk{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []ChatChunkChoice{{
			Index:        index,
			Delta:        delta,
			FinishReason: finishReason,
		}},
	}
}

func completionsUsageToChatUsage(usage *CompletionsUsage) *ChatUsage {
	if usage == nil {
		return nil
	}
	return &ChatUsage{
		PromptTokens:        usage.PromptTokens,
		CompletionTokens:    usage.CompletionTokens,
		TotalTokens:         usage.TotalTokens,
		PromptTokensDetails: usage.PromptTokensDetails,
	}
}

func normalizeCompletionFinishReason(reason string) string {
	if reason == "" {
		return "stop"
	}
	return reason
}

func jsonMarshalString(value string) ([]byte, error) {
	return json.Marshal(value)
}

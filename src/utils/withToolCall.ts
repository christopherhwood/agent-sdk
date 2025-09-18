import type { ToolResultEntry } from '../types/agent.js';
import { LogCategory } from '../types/logger.js';
import type { ToolCall, SessionState } from '../types/model.js';
import type { ToolResult, ToolFeedback } from '../types/tool-result.js';
import { LastToolError } from '../types/tool-result.js';
import type { ToolContext } from '../types/tool.js';

/**
 * Executes a tool call and guarantees that a matching `tool_result` block is
 * added to the conversation history – even on error or abort.
 *
 * It also appends an entry to the in‑memory `toolResults` array that the
 * AgentRunner uses for its cumulative result.
 * @param toolCall
 * @param sessionState
 * @param toolResults
 * @param exec
 * @param context
 * @param getToolFeedback Optional function that can return additional information about the tool result
 */
export async function withToolCall(
  toolCall: ToolCall,
  sessionState: SessionState,
  toolResults: ToolResultEntry[],
  exec: (ctx: ToolContext) => Promise<ToolResult>,
  context: ToolContext,
  getToolFeedback?: (result: ToolResult) => Promise<ToolFeedback | void>,
): Promise<ToolResult> {
  context.logger?.debug(
    `[withToolCall] Executing tool ${toolCall.toolId}, abortSignal=${context.abortSignal?.aborted}`,
    LogCategory.TOOLS,
  );
  let result: ToolResult;
  let aborted = false;

  try {
    try {
      const execPromise = exec(context);

      // If an abortSignal is provided, race the execution against it so we can
      // resolve promptly when the caller aborts – even if the underlying tool
      // ignores the signal.
      if (context.abortSignal) {
        result = (await Promise.race([
          execPromise,
          new Promise<unknown>((_, reject) => {
            const onAbort = () => {
              context.logger?.debug(
                `[withToolCall] AbortSignal 'abort' event received`,
                LogCategory.TOOLS,
              );
              context.abortSignal!.removeEventListener('abort', onAbort);
              reject(new Error('AbortError'));
            };
            if (context.abortSignal!.aborted) {
              context.logger?.debug(
                `[withToolCall] AbortSignal was already aborted when Promise.race started`,
                LogCategory.TOOLS,
              );
              return onAbort();
            }
            context.abortSignal!.addEventListener('abort', onAbort);
          }),
        ])) as ToolResult;
      } else {
        result = await execPromise;
      }
    } catch (err) {
      if ((err as Error).message === 'AbortError') {
        context.logger?.debug(
          `[withToolCall] Caught AbortError, marking result as aborted`,
          LogCategory.TOOLS,
        );
        aborted = true;
        // Return a typed error result for aborts
        result = { ok: false, error: 'AbortError' };
      } else {
        result = { ok: false, error: String(err) };
      }
    }

    // --------------------------------------------------------------
    // Check if tool returned a typed error and set lastToolError
    // --------------------------------------------------------------
    if (!result.ok) {
      sessionState.lastToolError = {
        toolId: toolCall.toolId,
        error: result.error,
        args: toolCall.args as Record<string, unknown>,
      };
    } else {
      // Clear previous error on success
      delete sessionState.lastToolError;
    }

    // --------------------------------------------------------------
    // Get additional information from feedback function if provided
    // --------------------------------------------------------------
    if (getToolFeedback && !aborted) {
      try {
        const additionalInfo = await getToolFeedback(result);
        if (additionalInfo) {
          // Non-mutating enrichment to avoid touching tool return objects
          result = { ...result, additionalInformation: additionalInfo };
        }
      } catch (err) {
        context.logger?.warn(
          `[withToolCall] Failed to get tool feedback: ${err}`,
          LogCategory.TOOLS,
        );
      }
    }

    // --------------------------------------------------------------
    // Decide whether we should append the `tool_result` message.  If
    // a rollback occurred while the tool was executing the accompanying
    // `tool_use` message may have been removed from the ContextWindow.
    // Adding a result in that case would break the required ordering
    // (tool_use must be immediately followed by the corresponding
    // tool_result).
    // --------------------------------------------------------------

    let shouldAppendResult = true;

    if (toolCall.toolUseId) {
      const lastMsg = sessionState.contextWindow.peek();
      const firstBlock = Array.isArray(lastMsg?.anthropic.content)
        ? (lastMsg!.anthropic.content[0] as any)
        : undefined;

      const stillHasToolUse =
        firstBlock?.type === 'tool_use' && firstBlock.id === toolCall.toolUseId;

      if (stillHasToolUse) {
        sessionState.contextWindow.pushToolResult(toolCall.toolUseId, result);
      } else {
        // ContextWindow has been rolled back – skip appending the result.
        context.logger?.warn(
          `[withToolCall] Skipping tool_result for toolUseId=${toolCall.toolUseId} because context was rolled back`,
        );
        shouldAppendResult = false;
      }
    }

    if (shouldAppendResult) {
      toolResults.push({
        toolId: toolCall.toolId,
        args: toolCall.args as Record<string, unknown>,
        result,
        toolUseId: toolCall.toolUseId,
        aborted,
      });
    }

    if (aborted) throw new Error('AbortError');

    return result;
  } catch (e) {
    throw e;
  }
}

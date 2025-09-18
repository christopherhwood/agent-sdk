/**
 * Discriminated union types for tool execution results
 */

export interface ToolFeedback {
  text?: string;
  attachments?: Attachment[];
}

export type Attachment = ImageAttachment;

export interface ImageAttachment {
  kind: 'image';
  // Prefer workspace-relative path; provider may read and base64 it.
  path?: string;
  // Alternatively, direct base64 or URL can be supplied.
  dataBase64?: string;
  url?: string;
  mime?: 'image/png' | 'image/jpeg' | 'image/webp';
  width?: number;
  height?: number;
  alt?: string;
}

interface WithAdditionalInformation {
  additionalInformation?: ToolFeedback;
}

export interface ToolSuccess<Data = unknown> extends WithAdditionalInformation {
  ok: true;
  data: Data;
}

export interface ToolError extends WithAdditionalInformation {
  ok: false;
  error: string;
}

export type ToolResult<Data = unknown> = ToolSuccess<Data> | ToolError;

/**
 * Tool-Specific Data Types
 */

export interface FileEditToolData {
  path: string;
  displayPath?: string;
  originalContent: string;
  newContent: string;
}

export interface FileReadToolData {
  content: string;
  path: string;
  lineCount?: number;
}

export interface LSToolData {
  entries: Array<{
    name: string;
    isDirectory: boolean;
    size?: number;
    modified?: string;
  }>;
  path: string;
}

export interface BashToolData {
  stdout: string;
  stderr: string;
  exitCode: number;
  command: string;
}

export interface GrepToolData {
  pattern: string;
  path: string;
  results: Array<{
    file: string;
    line: number;
    content: string;
  }>;
  count: number;
  hasMore: boolean;
  truncated: boolean;
  totalMatches: number;
}

export interface GlobToolData {
  pattern: string;
  matches: string[];
  count: number;
  hasMore: boolean;
  truncated: boolean;
  totalMatches: number;
}

/**
 * Type Aliases for Each Tool
 */

export type FileEditToolResult = ToolResult<FileEditToolData>;
export type FileReadToolResult = ToolResult<FileReadToolData>;
export type LSToolResult = ToolResult<LSToolData>;
export type BashToolResult = ToolResult<BashToolData>;
export type GrepToolResult = ToolResult<GrepToolData>;
export type GlobToolResult = ToolResult<GlobToolData>;

/**
 * Type for tracking the last tool error in session state
 */
export interface LastToolError {
  toolId: string;
  error: string;
  args: Record<string, unknown>;
}

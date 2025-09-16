/**
 * Callback interfaces for agent lifecycle events
 * @module Callbacks
 */

import type {
  ProcessingStartedData,
  ProcessingCompletedData,
  ProcessingErrorData,
  EnvironmentStatusData,
  CheckpointData,
  PermissionData,
} from './events.js';
import type { ToolExecutionState } from './tool-execution/index.js';

/**
 * Dynamic data providers
 */
export interface DynamicDataCallbacks {
  /**
   * Callback to dynamically retrieve a remote environment ID
   * Required when using remote environment (type === 'remote')
   * @returns Promise resolving to remote environment ID
   */
  getRemoteId?: (sessionId: string) => Promise<string>;
}

/**
 * Lifecycle hook callbacks
 */
export interface LifecycleCallbacks {
  /**
   * Called when a query processing starts
   */
  onProcessingStarted?: (data: ProcessingStartedData) => void | Promise<void>;

  /**
   * Called when a query processing completes successfully
   */
  onProcessingCompleted?: (data: ProcessingCompletedData) => void | Promise<void>;

  /**
   * Called when a query processing encounters an error
   */
  onProcessingError?: (data: ProcessingErrorData) => void | Promise<void>;

  /**
   * Called when a query processing is aborted
   */
  onProcessingAborted?: ({ sessionId }: { sessionId: string }) => void | Promise<void>;

  /**
   * Called when a tool execution starts
   */
  onToolExecutionStarted?: (execution: ToolExecutionState) => void | Promise<void>;

  /**
   * Called when a tool execution completes successfully
   * Can optionally return additional information to be included in the tool result
   */
  onToolExecutionCompleted?: (execution: ToolExecutionState) => Promise<string | void>;

  /**
   * Called when a tool execution encounters an error
   */
  onToolExecutionError?: (execution: ToolExecutionState) => void | Promise<void>;

  /**
   * Called when the environment status changes
   */
  onEnvironmentStatusChanged?: (status: EnvironmentStatusData) => void | Promise<void>;

  /**
   * Called when a checkpoint is ready
   */
  onCheckpointReady?: (checkpoint: CheckpointData) => void;

  /**
   * Called when a permission is requested
   */
  onPermissionRequested?: (permission: PermissionData) => Promise<boolean>;
}

/**
 * Combined agent callbacks
 */
export type AgentCallbacks = DynamicDataCallbacks & LifecycleCallbacks;

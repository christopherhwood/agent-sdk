/**
 * Shared captioning interface that allows host applications to provide
 * their own image-to-text implementation.
 */
export interface ImageCaptionInput {
  /**
   * Original image reference. May be an HTTPS URL or a data URL.
   */
  urlOrData: string;
  /**
   * Optional session identifier for context or local lookup.
   */
  sessionId?: string;
}

export interface ImageCaptioner {
  /**
   * Convert one or more image references into textual captions.
   * The returned array must align with the order of `inputs`.
   */
  describeImages(inputs: ImageCaptionInput[]): Promise<string[]>;
}

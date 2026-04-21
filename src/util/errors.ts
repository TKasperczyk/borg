export type BorgErrorOptions = {
  cause?: unknown;
};

export type BorgTypedErrorOptions = BorgErrorOptions & {
  code?: string;
};

export type BorgErrorJSON = {
  name: string;
  code: string;
  message: string;
  cause?: unknown;
};

function serializeCause(cause: unknown): unknown {
  if (!(cause instanceof Error)) {
    return cause;
  }

  return {
    name: cause.name,
    message: cause.message,
  };
}

export abstract class BorgError extends Error {
  readonly code: string;

  constructor(code: string, message: string, options: BorgErrorOptions = {}) {
    super(message, { cause: options.cause });
    this.name = new.target.name;
    this.code = code;
  }

  toJSON(): BorgErrorJSON {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      cause: serializeCause(this.cause),
    };
  }
}

export class ConfigError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_CONFIG_ERROR", message, options);
  }
}

export class StreamError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_STREAM_ERROR", message, options);
  }
}

export class EmbeddingError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_EMBEDDING_ERROR", message, options);
  }
}

export class LLMError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_LLM_ERROR", message, options);
  }
}

export class StorageError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_STORAGE_ERROR", message, options);
  }
}

export class CognitionError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_COGNITION_ERROR", message, options);
  }
}

export class WorkingMemoryError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_WORKING_MEMORY_ERROR", message, options);
  }
}

export class SemanticError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_SEMANTIC_ERROR", message, options);
  }
}

export class CommitmentError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_COMMITMENT_ERROR", message, options);
  }
}

export class BudgetExceededError extends BorgError {
  constructor(message: string, options: BorgTypedErrorOptions = {}) {
    super(options.code ?? "BORG_BUDGET_EXCEEDED", message, options);
  }
}

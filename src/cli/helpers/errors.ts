// CLI-specific error type used to keep argument and not-found failures user-facing.
import { BorgError } from "../../util/errors.js";

export class CliError extends BorgError {
  constructor(message: string, options: { cause?: unknown; code?: string } = {}) {
    super(options.code ?? "CLI_ARGUMENT", message, options);
  }
}

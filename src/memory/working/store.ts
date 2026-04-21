import { existsSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";

import { readJsonFile, writeJsonFileAtomic } from "../../util/atomic-write.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { WorkingMemoryError } from "../../util/errors.js";
import type { SessionId } from "../../util/ids.js";

import {
  createWorkingMemory,
  pushRecentThought,
  type WorkingMemory,
  workingMemorySchema,
} from "./types.js";

export type WorkingMemoryStoreOptions = {
  dataDir?: string;
  clock?: Clock;
  maxRecentThoughts?: number;
};

function cloneWorkingMemory(state: WorkingMemory): WorkingMemory {
  return structuredClone(state) as WorkingMemory;
}

export class WorkingMemoryStore {
  private readonly clock: Clock;
  private readonly maxRecentThoughts: number;
  private readonly states = new Map<SessionId, WorkingMemory>();

  constructor(private readonly options: WorkingMemoryStoreOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
    this.maxRecentThoughts = options.maxRecentThoughts ?? 10;
  }

  private get workingDirectory(): string | undefined {
    return this.options.dataDir === undefined ? undefined : join(this.options.dataDir, "working");
  }

  private getPath(sessionId: SessionId): string {
    const workingDirectory = this.workingDirectory;

    if (workingDirectory === undefined) {
      throw new WorkingMemoryError("Working memory persistence is not configured", {
        code: "WORKING_MEMORY_PERSISTENCE_DISABLED",
      });
    }

    return join(workingDirectory, `${sessionId}.json`);
  }

  load(sessionId: SessionId): WorkingMemory {
    const cached = this.states.get(sessionId);

    if (cached !== undefined) {
      return cloneWorkingMemory(cached);
    }

    const persisted = this.readPersisted(sessionId);
    const state = persisted ?? createWorkingMemory(sessionId, this.clock.now());
    this.states.set(sessionId, cloneWorkingMemory(state));
    return cloneWorkingMemory(state);
  }

  save(state: WorkingMemory): WorkingMemory {
    const parsed = workingMemorySchema.safeParse({
      ...state,
      recent_thoughts: state.recent_thoughts.slice(-this.maxRecentThoughts),
    });

    if (!parsed.success) {
      throw new WorkingMemoryError("Invalid working memory state", {
        cause: parsed.error,
        code: "WORKING_MEMORY_INVALID",
      });
    }

    const next = cloneWorkingMemory(parsed.data);
    this.writePersisted(next);
    this.states.set(next.session_id, next);
    return cloneWorkingMemory(next);
  }

  clear(sessionId: SessionId): void {
    this.states.delete(sessionId);

    const workingDirectory = this.workingDirectory;

    if (workingDirectory === undefined) {
      return;
    }

    const filePath = join(workingDirectory, `${sessionId}.json`);

    if (!existsSync(filePath)) {
      return;
    }

    try {
      rmSync(filePath, { force: true });
    } catch (error) {
      throw new WorkingMemoryError(`Failed to clear working memory for ${sessionId}`, {
        cause: error,
        code: "WORKING_MEMORY_CLEAR_FAILED",
      });
    }
  }

  addRecentThought(sessionId: SessionId, thought: string): WorkingMemory {
    const current = this.load(sessionId);
    return this.save(pushRecentThought(current, thought, this.maxRecentThoughts));
  }

  private readPersisted(sessionId: SessionId): WorkingMemory | undefined {
    if (this.workingDirectory === undefined) {
      return undefined;
    }

    try {
      const raw = readJsonFile<unknown>(this.getPath(sessionId));

      if (raw === undefined) {
        return undefined;
      }

      const parsed = workingMemorySchema.safeParse(raw);

      if (!parsed.success) {
        throw new WorkingMemoryError(`Invalid working memory file for ${sessionId}`, {
          cause: parsed.error,
          code: "WORKING_MEMORY_INVALID",
        });
      }

      return parsed.data;
    } catch (error) {
      if (error instanceof WorkingMemoryError) {
        throw error;
      }

      throw new WorkingMemoryError(`Failed to load working memory for ${sessionId}`, {
        cause: error,
        code: "WORKING_MEMORY_LOAD_FAILED",
      });
    }
  }

  private writePersisted(state: WorkingMemory): void {
    if (this.workingDirectory === undefined) {
      return;
    }

    try {
      mkdirSync(this.workingDirectory, { recursive: true });
      writeJsonFileAtomic(this.getPath(state.session_id), state);
    } catch (error) {
      throw new WorkingMemoryError(`Failed to save working memory for ${state.session_id}`, {
        cause: error,
        code: "WORKING_MEMORY_SAVE_FAILED",
      });
    }
  }
}

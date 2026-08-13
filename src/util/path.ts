import { homedir } from "node:os";
import { lstatSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";

import { isNodeError } from "./guards.js";

function pathEntryExists(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return false;
    }
    throw error;
  }
}

export function expandPath(pathLike: string): string {
  if (pathLike === "~") {
    return homedir();
  }

  if (pathLike.startsWith("~/")) {
    return join(homedir(), pathLike.slice(2));
  }

  return isAbsolute(pathLike) ? pathLike : resolve(pathLike);
}

/** Resolve symlinks in every existing ancestor while allowing a new leaf path. */
export function resolveRealPathForCreation(pathLike: string): string {
  const absolute = resolve(pathLike);
  const missingSegments: string[] = [];
  let cursor = absolute;

  while (!pathEntryExists(cursor)) {
    const parent = dirname(cursor);
    if (parent === cursor) {
      break;
    }
    missingSegments.unshift(cursor.slice(parent.length + (parent.endsWith(sep) ? 0 : 1)));
    cursor = parent;
  }

  return resolve(realpathSync(cursor), ...missingSegments);
}

export function isPathWithin(parentPath: string, candidatePath: string): boolean {
  const child = relative(parentPath, candidatePath);
  return child === "" || (!child.startsWith(`..${sep}`) && child !== ".." && !isAbsolute(child));
}

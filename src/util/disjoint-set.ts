export type DisjointSetRootComparator<T> = (leftRoot: T, rightRoot: T) => number;

export class DisjointSet<T> {
  private readonly parents: Map<T, T>;

  constructor(
    private readonly compareRoots: DisjointSetRootComparator<T>,
    parents?: Map<T, T>,
  ) {
    this.parents = parents ?? new Map<T, T>();
  }

  add(value: T): void {
    if (!this.parents.has(value)) {
      this.parents.set(value, value);
    }
  }

  find(value: T): T {
    let root = this.parents.get(value);

    if (root === undefined) {
      this.parents.set(value, value);
      return value;
    }

    while (root !== (this.parents.get(root) ?? root)) {
      root = this.parents.get(root) ?? root;
    }

    let current = value;
    while (current !== root) {
      const parent = this.parents.get(current);
      this.parents.set(current, root);

      if (parent === undefined || parent === current) {
        break;
      }

      current = parent;
    }

    return root;
  }

  union(left: T, right: T): void {
    const leftRoot = this.find(left);
    const rightRoot = this.find(right);

    if (leftRoot === rightRoot) {
      return;
    }

    if (this.compareRoots(leftRoot, rightRoot) <= 0) {
      this.parents.set(rightRoot, leftRoot);
      return;
    }

    this.parents.set(leftRoot, rightRoot);
  }
}

import { runRepairCli } from "./audited-data-repair.js";
import { contradictionRepairCli } from "./contradiction-edge-repair.js";

runRepairCli(import.meta.url, () =>
  contradictionRepairCli("backfill-superseded-contradiction-edges"),
);

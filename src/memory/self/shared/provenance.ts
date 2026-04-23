import { provenanceSchema, type Provenance } from "../../common/provenance.js";
import { ProvenanceError } from "../../../util/errors.js";

export function requireProvenance(provenance: Provenance | undefined, label: string): Provenance {
  if (provenance === undefined) {
    throw new ProvenanceError(`${label} requires provenance`, {
      code: "PROVENANCE_REQUIRED",
    });
  }

  return provenanceSchema.parse(provenance);
}

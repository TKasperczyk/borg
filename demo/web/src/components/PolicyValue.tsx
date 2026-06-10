import type {
  CreatorDirectiveActivationScope,
  CreatorDirectiveContentScope,
  CreatorDirectiveMentionPolicy,
  SessionParticipationPolicy,
  SessionPrivacyLevel,
} from "../api/types";
import { displayValue, shortId } from "../screens/screen-utils";
import { IdRef } from "./Inspector/IdRef";
import { Tag, type TagKind } from "./Tag";

type ScalarPolicyDomain =
  | "content_scope"
  | "activation_scope"
  | "mention_policy"
  | "privacy_level"
  | "participation_policy";

export type PolicyValueDomain = ScalarPolicyDomain | "entity-list";

type EntityListMode = "allowed" | "excluded";

type PolicyValueProps =
  | {
      domain: ScalarPolicyDomain;
      value: unknown;
    }
  | {
      domain: "entity-list";
      value: unknown;
      mode: EntityListMode;
    };

function scalarText(value: unknown): string {
  if (typeof value === "string" && value.length > 0) {
    return value;
  }
  return "-";
}

function scalarKind(domain: ScalarPolicyDomain, value: unknown): TagKind {
  if (domain === "content_scope") {
    switch (value as CreatorDirectiveContentScope) {
      case "public":
        return "info";
      case "allow_list":
      case "subject_only":
        return "purple";
      case "operator_only":
      case "all_except":
        return "warn";
      default:
        return "solid";
    }
  }

  if (domain === "activation_scope") {
    switch (value as CreatorDirectiveActivationScope) {
      case "public":
      case "same_as_disclosure":
        return "info";
      case "allow_list":
      case "subject_only":
        return "purple";
      case "operator_only":
      case "all_except":
        return "warn";
      default:
        return "solid";
    }
  }

  if (domain === "mention_policy") {
    switch (value as CreatorDirectiveMentionPolicy) {
      case "proactive":
        return "acc";
      case "answer_if_asked":
      case "only_if_topic_raised":
        return "";
      case "never_mention":
        return "solid";
      default:
        return "solid";
    }
  }

  if (domain === "privacy_level") {
    switch (value as SessionPrivacyLevel) {
      case "payload_off":
        return "info";
      case "payload_on":
        return "warn";
      default:
        return "solid";
    }
  }

  switch (value as SessionParticipationPolicy) {
    case "active":
      return "acc";
    case "observing":
      return "info";
    case "paused":
      return "warn";
    case "muted":
      return "solid";
    default:
      return "solid";
  }
}

function entityIds(value: unknown): string[] | null {
  if (typeof value === "string") {
    return value.length === 0 ? [] : [value];
  }
  if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
    return value;
  }
  return null;
}

function EntityListPolicyValue({ value, mode }: { value: unknown; mode: EntityListMode }) {
  const ids = entityIds(value);

  if (ids === null) {
    return (
      <span className="policy-value dim" data-policy-domain="entity-list">
        {displayValue(value)}
      </span>
    );
  }

  const label = `${ids.length} ${ids.length === 1 ? "entity" : "entities"}`;

  return (
    <span className="policy-value" data-policy-domain="entity-list">
      <Tag kind={mode === "allowed" ? "info" : "warn"}>{label}</Tag>
      {ids.map((id, index) => (
        <span key={id}>
          {index === 0 ? " " : ", "}
          <IdRef id={id} type="entity" label={shortId(id)} />
        </span>
      ))}
    </span>
  );
}

export function PolicyValue(props: PolicyValueProps) {
  if (props.domain === "entity-list") {
    return <EntityListPolicyValue value={props.value} mode={props.mode} />;
  }

  return (
    <span className="policy-value" data-policy-domain={props.domain}>
      <Tag kind={scalarKind(props.domain, props.value)}>{scalarText(props.value)}</Tag>
    </span>
  );
}

// Store-leader configurable rules for each item × daypart.
// Sprint 1: all defaults (no DO NOT COOK, no min presentation).
// Sprint 2: expose a settings UI that populates and persists StoreConfig.

export type DaypartRule = {
  doNotCook: boolean;      // Sprint 2: store-leader configurable per item × daypart
  minPresentation: number; // Sprint 2: store-leader configurable (units, already LCU-aligned)
};

export type ItemRules = {
  dayparts: Record<number, DaypartRule>; // key = daypart start hour (0–23)
};

export type StoreConfig = {
  items: Record<string, ItemRules | undefined>; // keyed by OvenItemId string
};

// Sprint 1 default: no restrictions, no minimum presentations anywhere.
export const DEFAULT_STORE_CONFIG: StoreConfig = { items: {} };

export function getDaypartRule(
  config: StoreConfig,
  id: string,
  daypartHour: number,
): DaypartRule {
  return (
    config.items[id]?.dayparts[daypartHour] ?? { doNotCook: false, minPresentation: 0 }
  );
}

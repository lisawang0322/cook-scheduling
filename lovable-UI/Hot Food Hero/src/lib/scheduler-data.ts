export type StoreType = "Urban" | "Suburban" | "Highway";

export interface FoodItem {
  id: string;
  name: string;
  category: "Oven" | "Roller Grill" | "Display";
  forecastDemand: number;
  batchSize: number;
  cookTimeMin: number;
  holdTimeHours: number;
  timeRemainingHours: number;
  wasteRate: number; // historical % at this store/time
  icon: string; // emoji
}

export interface Scenario {
  id: string;
  title: string;
  storeType: StoreType;
  day: string;
  time: string;
  items: FoodItem[];
}

export const SCENARIOS: Scenario[] = [
  {
    id: "A",
    title: "11:47 AM Lunch Rush",
    storeType: "Urban",
    day: "Friday",
    time: "11:47 AM",
    items: [
      { id: "pizza", name: "Pizza", category: "Oven", forecastDemand: 12, batchSize: 6, cookTimeMin: 15, holdTimeHours: 2, timeRemainingHours: 1.2, wasteRate: 0.18, icon: "🍕" },
      { id: "wings2", name: "Wings (2hr hold)", category: "Oven", forecastDemand: 10, batchSize: 5, cookTimeMin: 12, holdTimeHours: 2, timeRemainingHours: 1.2, wasteRate: 0.34, icon: "🍗" },
      { id: "wings4", name: "Wings (4hr hold)", category: "Oven", forecastDemand: 16, batchSize: 8, cookTimeMin: 14, holdTimeHours: 4, timeRemainingHours: 3.2, wasteRate: 0.22, icon: "🍗" },
      { id: "baked", name: "Baked Goods", category: "Oven", forecastDemand: 8, batchSize: 8, cookTimeMin: 22, holdTimeHours: 24, timeRemainingHours: 18.2, wasteRate: 0.08, icon: "🥐" },
      { id: "taquitos", name: "Taquitos", category: "Roller Grill", forecastDemand: 14, batchSize: 7, cookTimeMin: 10, holdTimeHours: 4, timeRemainingHours: 3.2, wasteRate: 0.12, icon: "🌯" },
    ],
  },
  {
    id: "B",
    title: "6:00 AM Morning Start",
    storeType: "Suburban",
    day: "Monday",
    time: "6:00 AM",
    items: [
      { id: "baked", name: "Baked Goods", category: "Oven", forecastDemand: 33, batchSize: 8, cookTimeMin: 22, holdTimeHours: 24, timeRemainingHours: 24, wasteRate: 0.06, icon: "🥐" },
      { id: "pizza", name: "Pizza", category: "Oven", forecastDemand: 6, batchSize: 6, cookTimeMin: 15, holdTimeHours: 2, timeRemainingHours: 2, wasteRate: 0.14, icon: "🍕" },
      { id: "taquitos", name: "Taquitos", category: "Roller Grill", forecastDemand: 4, batchSize: 4, cookTimeMin: 10, holdTimeHours: 4, timeRemainingHours: 4, wasteRate: 0.1, icon: "🌯" },
    ],
  },
  {
    id: "C",
    title: "10:15 PM Late Night",
    storeType: "Highway",
    day: "Saturday",
    time: "10:15 PM",
    items: [
      { id: "wings2", name: "Wings (2hr hold)", category: "Oven", forecastDemand: 5, batchSize: 5, cookTimeMin: 12, holdTimeHours: 2, timeRemainingHours: 0.8, wasteRate: 0.4, icon: "🍗" },
      { id: "pizza", name: "Pizza", category: "Oven", forecastDemand: 6, batchSize: 6, cookTimeMin: 15, holdTimeHours: 2, timeRemainingHours: 1.8, wasteRate: 0.2, icon: "🍕" },
    ],
  },
];

// ML priority score: higher = cook sooner
export function priorityScore(item: FoodItem, demandMultiplier = 1): number {
  const urgency = 1 / Math.max(item.timeRemainingHours, 0.1); // expires sooner -> higher
  const demandDensity = (item.forecastDemand * demandMultiplier) / Math.max(item.holdTimeHours, 0.5);
  const wastePenalty = item.wasteRate * 2;
  return urgency * 3 + demandDensity * 0.4 + wastePenalty * 2;
}

export function rankItems(items: FoodItem[], demandMultiplier = 1): FoodItem[] {
  return [...items].sort((a, b) => priorityScore(b, demandMultiplier) - priorityScore(a, demandMultiplier));
}

export function explainRank(item: FoodItem, rank: number): string {
  if (rank === 0) {
    return `Recommended first — window ends in ${item.timeRemainingHours.toFixed(1)}h and historical waste rate is ${(item.wasteRate * 100).toFixed(0)}% at this store/daypart.`;
  }
  if (rank === 1) {
    return `Queued next — demand of ${item.forecastDemand} units fits a fresh batch right after the leading cook.`;
  }
  return `Lower priority — ${item.holdTimeHours}h hold window gives headroom; cook after higher-urgency items clear the oven.`;
}

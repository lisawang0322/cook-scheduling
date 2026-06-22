import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

// Resolved at request time so the env var can be set at runtime.
function pythonApi() {
  return (
    (typeof process !== "undefined" && process.env.PYTHON_API_URL) ||
    "http://localhost:8000"
  );
}

async function pyFetch(path: string, init?: RequestInit): Promise<unknown> {
  const res = await fetch(`${pythonApi()}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Python API ${path} → ${res.status}: ${text}`);
  }
  return res.json();
}

// ---- /api/rank ----

const ItemInputSchema = z.object({
  id: z.string(),
  forecast_demand: z.number(),
  lcu: z.number(),
  hold_time: z.number(),
  time_remaining: z.number(),
});

export const RankRequestSchema = z.object({
  store_type: z.string(),
  day_of_week: z.string(),
  is_weekend: z.boolean(),
  decision_hour: z.number(),
  items: z.array(ItemInputSchema),
});

export type RankRequest = z.infer<typeof RankRequestSchema>;

export type RankResponse = {
  v22_ranking: string[];
  explanations: string[];
};

export const getRankings = createServerFn({ method: "POST" })
  .inputValidator(RankRequestSchema)
  .handler(async ({ data }): Promise<RankResponse> => {
    return pyFetch("/api/rank", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }) as Promise<RankResponse>;
  });

// ---- /api/metrics ----

export type MetricsResponse = {
  total_scenarios: number;
  associate_accuracy: number;
  v1_accuracy: number;
  v22_accuracy: number;
  by_store: Record<string, { associate: number; v1: number; v22: number; n: number }>;
  by_hour: Record<string, { associate: number; v1: number; v22: number; n: number }>;
};

export const getMetrics = createServerFn({ method: "GET" }).handler(
  async (): Promise<MetricsResponse> => {
    return pyFetch("/api/metrics") as Promise<MetricsResponse>;
  },
);

// ---- /api/log-action ----

export const logAction = createServerFn({ method: "POST" })
  .inputValidator(z.record(z.unknown()))
  .handler(async ({ data }) => {
    return pyFetch("/api/log-action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  });

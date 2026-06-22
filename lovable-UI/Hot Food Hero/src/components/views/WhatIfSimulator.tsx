import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sliders, TrendingUp } from "lucide-react";
import { priorityScore, type FoodItem, type StoreType } from "@/lib/scheduler-data";

const STORE_MULT: Record<StoreType, number> = { Urban: 1.4, Suburban: 1.0, Highway: 0.7 };

const BASE_ITEMS: FoodItem[] = [
  { id: "pizza", name: "Pizza", category: "Oven", forecastDemand: 8, batchSize: 6, cookTimeMin: 15, holdTimeHours: 2, timeRemainingHours: 1.5, wasteRate: 0.18, icon: "🍕" },
  { id: "wings2", name: "Wings (2hr)", category: "Oven", forecastDemand: 6, batchSize: 5, cookTimeMin: 12, holdTimeHours: 2, timeRemainingHours: 1.3, wasteRate: 0.32, icon: "🍗" },
  { id: "wings4", name: "Wings (4hr)", category: "Oven", forecastDemand: 12, batchSize: 8, cookTimeMin: 14, holdTimeHours: 4, timeRemainingHours: 3, wasteRate: 0.2, icon: "🍗" },
  { id: "baked", name: "Baked Goods", category: "Oven", forecastDemand: 8, batchSize: 8, cookTimeMin: 22, holdTimeHours: 24, timeRemainingHours: 18, wasteRate: 0.08, icon: "🥐" },
];

function timeMultiplier(hour: number) {
  // lunch peak 11-13, dinner peak 17-19
  if (hour >= 11 && hour <= 13) return 1.5;
  if (hour >= 17 && hour <= 19) return 1.35;
  if (hour >= 6 && hour <= 9) return 1.15;
  return 0.8;
}

export function WhatIfSimulator() {
  const [storeType, setStoreType] = useState<StoreType>("Urban");
  const [hour, setHour] = useState(12);
  const [isWeekend, setIsWeekend] = useState(true);
  const [demand, setDemand] = useState<Record<string, number>>(() =>
    Object.fromEntries(BASE_ITEMS.map((i) => [i.id, i.forecastDemand]))
  );

  const multiplier = STORE_MULT[storeType] * timeMultiplier(hour) * (isWeekend ? 1.3 : 1);

  const items: FoodItem[] = useMemo(
    () => BASE_ITEMS.map((i) => ({ ...i, forecastDemand: demand[i.id] })),
    [demand]
  );

  const ranked = useMemo(
    () => [...items].map((it) => ({ it, score: priorityScore(it, multiplier) })).sort((a, b) => b.score - a.score),
    [items, multiplier]
  );

  const topName = ranked[0].it.name;
  const promotionMsg =
    multiplier > 1.4 && topName.startsWith("Pizza")
      ? "Pizza promoted to Rank 1 due to weekend lunch peak demand density."
      : `${topName} held at Rank 1 — urgency × waste-rate signal dominates.`;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3"><Sliders className="h-7 w-7 text-primary" /> What-If Simulator</h1>
        <p className="text-muted-foreground mt-1">Tune store conditions and watch the ML priority queue recompute live.</p>
      </div>

      <div className="grid lg:grid-cols-[360px_1fr] gap-6">
        <Card>
          <CardHeader><CardTitle>Controls</CardTitle></CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Lbl>Store Type</Lbl>
              <Select value={storeType} onValueChange={(v) => setStoreType(v as StoreType)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="Urban">Urban (×1.4 demand)</SelectItem>
                  <SelectItem value="Suburban">Suburban (baseline)</SelectItem>
                  <SelectItem value="Highway">Highway (×0.7)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between"><Lbl>Time of Day</Lbl><span className="text-sm font-bold tabular-nums">{hour}:00</span></div>
              <Slider min={6} max={23} step={1} value={[hour]} onValueChange={(v) => setHour(v[0])} />
              <div className="flex justify-between text-xs text-muted-foreground"><span>6 AM</span><span>11 PM</span></div>
            </div>

            <div className="flex items-center justify-between rounded-lg border p-3">
              <div>
                <Lbl>Weekend Boost</Lbl>
                <div className="text-xs text-muted-foreground">+30% demand</div>
              </div>
              <Switch checked={isWeekend} onCheckedChange={setIsWeekend} />
            </div>

            <div className="space-y-4 pt-2 border-t">
              <Lbl>Item Demand</Lbl>
              {BASE_ITEMS.map((it) => (
                <div key={it.id} className="space-y-1.5">
                  <div className="flex justify-between text-sm">
                    <span>{it.icon} {it.name}</span>
                    <b className="tabular-nums">{demand[it.id]}</b>
                  </div>
                  <Slider min={0} max={30} step={1} value={[demand[it.id]]} onValueChange={(v) => setDemand((d) => ({ ...d, [it.id]: v[0] }))} />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card className="border-primary/30 border-2">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Live Cook Sequence</CardTitle>
                <Badge className="bg-primary text-primary-foreground gap-1.5"><TrendingUp className="h-3 w-3" /> Demand × {multiplier.toFixed(2)}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {ranked.map(({ it, score }, idx) => (
                <div
                  key={it.id}
                  className="flex items-center gap-4 p-4 rounded-xl border bg-card transition-all duration-500"
                  style={{ borderLeftWidth: 4, borderLeftColor: idx === 0 ? "var(--red-brand)" : idx === 1 ? "var(--orange)" : "var(--primary)" }}
                >
                  <div className="text-2xl font-bold tabular-nums w-8 text-muted-foreground">#{idx + 1}</div>
                  <div className="text-3xl">{it.icon}</div>
                  <div className="flex-1">
                    <div className="font-semibold">{it.name}</div>
                    <div className="text-xs text-muted-foreground">Demand {it.forecastDemand} · Hold {it.holdTimeHours}h · Expires {it.timeRemainingHours}h</div>
                  </div>
                  <Badge variant="outline" className="tabular-nums">Score {score.toFixed(2)}</Badge>
                </div>
              ))}
              <div className="rounded-lg bg-primary/5 border border-primary/20 p-3 text-sm text-primary">
                👉 {promotionMsg}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-sm">Active Formula Values</CardTitle></CardHeader>
            <CardContent className="grid sm:grid-cols-3 gap-3 text-sm">
              <Formula label="Urgency" value={`1 / time_remaining`} highlight={`max ${(1 / Math.max(...items.map((i) => 1)) * 1).toFixed(2)}`} />
              <Formula label="Demand Density" value={`demand × ${multiplier.toFixed(2)} / hold_hours`} />
              <Formula label="Waste Penalty" value={`waste_rate × 2`} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function Lbl({ children }: { children: React.ReactNode }) {
  return <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">{children}</div>;
}

function Formula({ label, value, highlight }: { label: string; value: string; highlight?: string }) {
  return (
    <div className="rounded-lg border bg-muted/30 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-xs mt-1">{value}</div>
      {highlight && <div className="text-xs mt-1 text-primary font-semibold">{highlight}</div>}
    </div>
  );
}

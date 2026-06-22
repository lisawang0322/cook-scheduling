import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { XCircle, AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import { SCENARIOS, type FoodItem } from "@/lib/scheduler-data";
import { getRankings, type RankRequest } from "@/lib/api/scheduler.functions";

const METHODS = [
  {
    key: "associate",
    title: "Simulated Associate",
    accuracy: 55,
    waste: 42.5,
    completion: 62,
    Icon: XCircle,
    tone: "text-red-brand",
    border: "border-red-brand/40",
    chip: "bg-red-brand/10 text-red-brand border-red-brand/30",
    verdict: "High Waste Risk",
    detail: "Defaults to biggest-volume item (baked goods, 33 units) and ignores window expiry. Pizza expires in the warmer; wings under-cooked.",
  },
  {
    key: "rules",
    title: "Rule-Based v1",
    accuracy: 70,
    waste: 22.1,
    completion: 81,
    Icon: AlertTriangle,
    tone: "text-orange",
    border: "border-orange/40",
    chip: "bg-orange/10 text-orange border-orange/30",
    verdict: "Moderate Outcome",
    detail: "Priority-score formula avoids extreme deadlines, but misses hour-specific volume curves and store-type variance.",
  },
  {
    key: "ml",
    title: "ML v2.2 Optimizer",
    accuracy: 74.3,
    waste: 8.4,
    completion: 96,
    Icon: CheckCircle2,
    tone: "text-success",
    border: "border-success/50",
    chip: "bg-success/10 text-success border-success/30",
    verdict: "Optimal Outcome — Winner",
    detail: "Pairwise model with temporal aggregation. Sequences high-waste wings before pizza based on store history.",
  },
];

const ITEM_ID_MAP: Record<string, string> = {
  pizza: "pizza",
  wings2: "wings_2h",
  wings4: "wings_4h",
  baked: "baked_goods",
};

const ITEM_DISPLAY: Record<string, string> = {
  pizza: "🍕 Pizza",
  wings_2h: "🍗 Wings (2hr)",
  wings_4h: "🍗 Wings (4hr)",
  baked_goods: "🧁 Baked Goods",
};

function parseHour(time: string): number {
  const [hhmm, period] = time.split(" ");
  let [h] = hhmm.split(":").map(Number);
  if (period === "PM" && h !== 12) h += 12;
  if (period === "AM" && h === 12) h = 0;
  return h;
}

function buildRankRequest(items: FoodItem[], storeType: string, day: string, time: string): RankRequest {
  const WEEKEND = new Set(["Saturday", "Sunday"]);
  return {
    store_type: storeType.toLowerCase(),
    day_of_week: day,
    is_weekend: WEEKEND.has(day),
    decision_hour: parseHour(time),
    items: items
      .filter((item) => ITEM_ID_MAP[item.id])
      .map((item) => ({
        id: ITEM_ID_MAP[item.id],
        forecast_demand: item.forecastDemand,
        lcu: item.batchSize,
        hold_time: item.holdTimeHours,
        time_remaining: item.timeRemainingHours,
      })),
  };
}

export function ScenarioComparison() {
  const [scenarioId, setScenarioId] = useState(SCENARIOS[0].id);
  const scenario = SCENARIOS.find((s) => s.id === scenarioId)!;
  const maxWaste = Math.max(...METHODS.map((m) => m.waste));

  const rankRequest = useMemo(
    () => buildRankRequest(scenario.items, scenario.storeType, scenario.day, scenario.time),
    [scenarioId], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const { data: rankings, isLoading: rankLoading } = useQuery({
    queryKey: ["rankings", scenarioId],
    queryFn: () => getRankings({ data: rankRequest }),
    staleTime: Infinity,
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Scenario Comparison</h1>
          <p className="text-muted-foreground mt-1">Compare three scheduling strategies on the same store snapshot.</p>
        </div>
        <div className="space-y-1">
          <Label>Active Scenario</Label>
          <Select value={scenarioId} onValueChange={setScenarioId}>
            <SelectTrigger className="w-[320px]"><SelectValue /></SelectTrigger>
            <SelectContent>
              {SCENARIOS.map((s) => (
                <SelectItem key={s.id} value={s.id}>Scenario {s.id} · {s.title} ({s.storeType})</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <Card className="bg-muted/40">
        <CardContent className="p-4 text-sm flex flex-wrap gap-x-6 gap-y-2">
          <span><span className="text-muted-foreground">Store:</span> <b>{scenario.storeType}</b></span>
          <span><span className="text-muted-foreground">Day:</span> <b>{scenario.day}</b></span>
          <span><span className="text-muted-foreground">Time:</span> <b>{scenario.time}</b></span>
          <span><span className="text-muted-foreground">Items in queue:</span> <b>{scenario.items.length}</b></span>
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-3 gap-4">
        {METHODS.map((m) => (
          <Card key={m.key} className={`card-lift border-2 ${m.border}`}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{m.title}</CardTitle>
                <m.Icon className={`h-6 w-6 ${m.tone}`} />
              </div>
              <Badge className={`${m.chip} border w-fit`}>{m.accuracy}% Accuracy</Badge>
            </CardHeader>
            <CardContent className="space-y-4">
              <Badge className={`${m.chip} border w-full justify-center py-2 text-sm`}>{m.verdict}</Badge>
              <p className="text-sm text-muted-foreground leading-relaxed">{m.detail}</p>
              <div>
                <div className="flex justify-between text-xs mb-1"><span>Projected Waste</span><b className={m.tone}>${m.waste.toFixed(2)}</b></div>
                <Progress value={(m.waste / maxWaste) * 100} className="h-2" />
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1"><span>Window Completion</span><b>{m.completion}%</b></div>
                <Progress value={m.completion} className="h-2" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader><CardTitle>Metrics Bar</CardTitle></CardHeader>
        <CardContent className="space-y-5">
          {METHODS.map((m) => (
            <div key={m.key}>
              <div className="flex justify-between text-sm mb-1">
                <span className="font-medium">{m.title}</span>
                <span className="text-muted-foreground">${m.waste.toFixed(2)} waste · {m.completion}% completion</span>
              </div>
              <div className="flex gap-2">
                <div className="h-3 rounded-full bg-muted flex-1 overflow-hidden">
                  <div className={`h-full ${m.key === "ml" ? "bg-success" : m.key === "rules" ? "bg-orange" : "bg-red-brand"} transition-all`} style={{ width: `${(m.waste / maxWaste) * 100}%` }} />
                </div>
                <div className="h-3 rounded-full bg-muted w-1/2 overflow-hidden">
                  <div className="h-full bg-primary transition-all" style={{ width: `${m.completion}%` }} />
                </div>
              </div>
            </div>
          ))}
          <div className="flex gap-4 text-xs text-muted-foreground pt-2 border-t">
            <span className="flex items-center gap-1.5"><div className="h-2 w-4 rounded bg-red-brand" /> Waste $</span>
            <span className="flex items-center gap-1.5"><div className="h-2 w-4 rounded bg-primary" /> Completion %</span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <div>
              <CardTitle>Live Rankings from Python Backend</CardTitle>
              <p className="text-xs text-muted-foreground mt-1">
                Real v1 rules · v2.2 ML · associate picks — computed from scenario features
              </p>
            </div>
            {rankLoading && (
              <span className="inline-flex items-center gap-1.5 text-xs text-primary">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Running models…
              </span>
            )}
            {rankings && (
              <Badge variant="outline" className="text-success border-success/40 bg-success/5 text-xs">
                Live · v2.2 ML
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {!rankings && !rankLoading && (
            <p className="text-sm text-muted-foreground">Backend unavailable — start uvicorn to see real rankings.</p>
          )}
          {rankings && (
            <div className="grid md:grid-cols-3 gap-4">
              {[
                {
                  label: "Associate (simulated)",
                  ranking: [rankings.associate_pick],
                  tone: "border-red-brand/40 bg-red-brand/5",
                  chip: "bg-red-brand/10 text-red-brand",
                },
                {
                  label: "v1 Rule-Based",
                  ranking: rankings.v1_ranking,
                  tone: "border-orange/40 bg-orange/5",
                  chip: "bg-orange/10 text-orange",
                },
                {
                  label: "v2.2 ML Optimizer",
                  ranking: rankings.v22_ranking,
                  tone: "border-success/40 bg-success/5",
                  chip: "bg-success/10 text-success",
                },
              ].map(({ label, ranking, tone, chip }) => (
                <div key={label} className={`rounded-xl border-2 p-4 ${tone} space-y-2`}>
                  <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{label}</div>
                  {ranking.map((id, idx) => (
                    <div key={id} className="flex items-center gap-2">
                      <Badge className={`${chip} border text-[10px] font-bold w-8 justify-center shrink-0`}>
                        #{idx + 1}
                      </Badge>
                      <span className="text-sm font-medium">{ITEM_DISPLAY[id] ?? id}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
          {rankings?.explanations && rankings.explanations.length > 0 && (
            <div className="mt-4 p-3 rounded-lg bg-muted/50 border text-sm text-muted-foreground leading-relaxed space-y-1">
              {rankings.explanations.map((ex, i) => (
                <p key={i} dangerouslySetInnerHTML={{ __html: ex.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>") }} />
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{children}</div>;
}

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Clock,
  Flame,
  Send,
  RotateCcw,
  Sparkles,
  Wand2,
  TrendingUp,
  Loader2,
} from "lucide-react";
import { getRankings, type RankRequest } from "@/lib/api/scheduler.functions";

import {
  computeScenario,
  generateForecasts,
  ITEM_DEFS,
  DEFAULT_SIM_INPUT,
  addMinutes,
  addHours,
  type SimulatorInput,
} from "@/lib/hot-food";
import { useScenario } from "@/lib/scenario-context";
import { toast } from "sonner";


const STORE_TYPES: SimulatorInput["storeType"][] = ["Urban", "Suburban", "Highway"];
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const DAY_FULL: Record<string, string> = {
  Mon: "Monday", Tue: "Tuesday", Wed: "Wednesday", Thu: "Thursday",
  Fri: "Friday", Sat: "Saturday", Sun: "Sunday",
};

const PRIORITY = [
  { label: "COOK NOW", chip: "bg-red-brand text-white", border: "border-l-red-brand" },
  { label: "NEXT",     chip: "bg-yellow-400 text-black", border: "border-l-yellow-400" },
  { label: "THEN",     chip: "bg-primary text-primary-foreground", border: "border-l-primary" },
];

export function ScenarioSimulator({ onSendToTablet }: { onSendToTablet: () => void }) {
  const { setActiveScenario, triggerNotification } = useScenario();
  const [input, setInput] = useState<SimulatorInput>(DEFAULT_SIM_INPUT);

  const scenario = useMemo(() => computeScenario(input), [input]);
  const forecasts = useMemo(() => generateForecasts(input), [input]);

  const rankRequest = useMemo((): RankRequest | null => {
    const included = forecasts.filter((f) => f.included);
    if (included.length === 0) return null;
    return {
      store_type: input.storeType.toLowerCase(),
      day_of_week: input.dayLabel,
      is_weekend: ["Saturday", "Sunday"].includes(input.dayLabel),
      decision_hour: input.hour,
      items: included.map((f) => ({
        id: f.id,
        forecast_demand: f.rounded,
        lcu: ITEM_DEFS[f.id].lcu,
        hold_time: ITEM_DEFS[f.id].holdTimeHr,
        time_remaining: ITEM_DEFS[f.id].holdTimeHr,
      })),
    };
  }, [forecasts, input.dayLabel, input.hour, input.storeType]);

  const {
    data: v22Result,
    isLoading: v22Loading,
    isError: v22Error,
  } = useQuery({
    queryKey: ["sim-rankings", rankRequest],
    queryFn: () => getRankings({ data: rankRequest! }),
    enabled: rankRequest !== null,
  });

  useEffect(() => {
    if (v22Error) {
      toast.error("ML backend unavailable", {
        description: "Showing rule-based preview until uvicorn app.api:app is running on :8000",
      });
    }
  }, [v22Error]);

  const displayScenario = useMemo(() => {
    if (!v22Result?.v22_ranking.length) return scenario;

    const itemById = new Map(scenario.items.map((item) => [item.id, item]));
    const orderedItems = v22Result.v22_ranking
      .map((id) => itemById.get(id))
      .filter((item): item is NonNullable<typeof item> => item !== undefined);

    for (const item of scenario.items) {
      if (!orderedItems.some((ordered) => ordered.id === item.id)) {
        orderedItems.push(item);
      }
    }

    return {
      ...scenario,
      items: orderedItems,
      reason: v22Result.explanations[0] ?? scenario.reason,
    };
  }, [scenario, v22Result]);

  const reset = () => setInput(DEFAULT_SIM_INPUT);

  const send = () => {
    setActiveScenario(displayScenario);
    triggerNotification();
    toast.success("Scenario sent to tablet", { description: "Switch to Associate Tablet — Step 1 will fire." });
    onSendToTablet();
  };

  const hourLabel = formatHour(input.hour);


  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
            <Wand2 className="h-3.5 w-3.5" /> Manager Console
          </div>
          <h1 className="text-2xl font-bold mt-1">Scenario Simulator</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            Configure store conditions, generate a prioritized cook list, then push it to the associate tablet.
          </p>
        </div>
        <Badge variant="outline" className="text-xs gap-1.5">
          {v22Loading && <Loader2 className="h-3 w-3 animate-spin" />}
          {v22Loading
            ? "Running ML v2.2…"
            : v22Result
              ? "v2.2 ML ranking"
              : v22Error
                ? "rule-based fallback"
                : "v2.2 ML ranking"}
        </Badge>
      </header>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* LEFT — Inputs */}
        <Card className="p-5 space-y-6">
          <section className="space-y-2">
            <Label className="text-xs uppercase tracking-wider text-muted-foreground">Store Type</Label>
            <div className="grid grid-cols-3 gap-2 p-1 bg-muted rounded-lg">
              {STORE_TYPES.map((t) => (
                <button
                  key={t}
                  onClick={() => setInput((p) => ({ ...p, storeType: t }))}
                  className={`py-2 text-sm font-medium rounded-md transition ${
                    input.storeType === t
                      ? "bg-white shadow text-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </section>

          <section className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-xs uppercase tracking-wider text-muted-foreground">Day of Week</Label>
              <Select
                value={Object.keys(DAY_FULL).find((k) => DAY_FULL[k] === input.dayLabel) ?? "Fri"}
                onValueChange={(v) => setInput((p) => ({ ...p, dayLabel: DAY_FULL[v] }))}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {DAYS.map((d) => (
                    <SelectItem key={d} value={d}>{DAY_FULL[d]}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label className="text-xs uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                <Clock className="h-3 w-3" /> Time of Day · {hourLabel}
              </Label>
              <Slider
                min={6}
                max={23}
                step={1}
                value={[input.hour]}
                onValueChange={([h]) => setInput((p) => ({ ...p, hour: h }))}
              />
            </div>
          </section>

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-xs uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                <TrendingUp className="h-3 w-3" /> Generated Forecast
              </Label>
              <Badge variant="outline" className="text-[10px]">auto · daypart qty</Badge>
            </div>
            <p className="text-[11px] text-muted-foreground -mt-1">
              Fractional hourly forecast allocated across dayparts using backend LCU rounding rules. Zero-allocated items are skipped.
            </p>
            <div className="space-y-2">
              {forecasts.map(({ id, rounded, included }: import("@/lib/hot-food").ForecastRow) => {
                const def = ITEM_DEFS[id];
                return (
                  <div
                    key={id}
                    className={`rounded-lg border p-3 flex items-center gap-3 transition ${
                      included ? "bg-white" : "bg-muted/30 opacity-70"
                    }`}
                  >
                    <div className="text-2xl">{def.emoji}</div>
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-sm leading-tight">{def.name}</div>
                      <div className="text-[11px] text-muted-foreground">
                        allocated qty · LCU {def.lcu} · {def.holdTimeHr}h hold
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold tabular-nums leading-none">{rounded}</div>
                      <div className="text-[10px] uppercase tracking-wider text-muted-foreground mt-0.5">
                        {included ? "cooking" : "skip"}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>


          <div className="flex gap-2 pt-2">
            <Button
              variant="outline"
              onClick={reset}
              className="flex-1"
            >
              <RotateCcw className="h-4 w-4 mr-1" /> Reset to Defaults
            </Button>
            <Button
              onClick={() => setInput((p) => ({ ...p }))}
              variant="secondary"
              className="flex-1"
            >
              <Sparkles className="h-4 w-4 mr-1" /> Generate Recommendation
            </Button>
          </div>
        </Card>

        {/* RIGHT — Live Preview */}
        <div className="space-y-4">
          <Card className="p-5">
            <div className="flex items-center justify-between gap-2">
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">Live Preview</div>
                <h2 className="text-lg font-bold leading-tight">Prioritized Cook List</h2>
              </div>
              <Badge className="bg-primary/10 text-primary border-primary/20" variant="outline">
                Store {scenario.storeId} · {scenario.storeType}
              </Badge>
            </div>

            <div className="mt-2 text-xs text-muted-foreground">
              {scenario.dayLabel} · {scenario.time}
            </div>

            <div className="mt-4 space-y-3">
              {displayScenario.items.length === 0 ? (
                <div className="text-sm text-muted-foreground p-6 text-center border rounded-lg">
                  No items included. Toggle on at least one oven item.
                </div>
              ) : v22Loading && !v22Result ? (
                <div className="text-sm text-muted-foreground p-6 text-center border rounded-lg flex items-center justify-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading ML v2.2 ranking…
                </div>
              ) : (
                displayScenario.items.map((item, idx) => {
                  const p = PRIORITY[Math.min(idx, 2)];
                  const ready = addMinutes(displayScenario.time, item.cookTimeMin);
                  const expiry = addHours(ready, item.holdTimeHr);
                  return (
                    <div
                      key={item.id}
                      className={`rounded-lg border bg-white border-l-4 ${p.border} p-3 flex items-start gap-3`}
                    >
                      <div className="text-2xl">{item.emoji}</div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge className={`${p.chip} text-[10px] font-bold tracking-wider`}>{p.label}</Badge>
                          <div className="font-semibold text-sm">{item.name}</div>
                        </div>
                        <div className="text-sm font-semibold mt-1">
                          Cook {item.recommended} units{" "}
                          <span className="text-xs text-muted-foreground font-normal">
                            ({Math.max(1, Math.round(item.recommended / item.lcu))} × {item.lcu})
                          </span>
                        </div>
                        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-muted-foreground">
                          <span>cook {item.cookTimeMin}m</span>
                          <span>ready {ready}</span>
                          <span>discard {expiry}</span>
                        </div>
                      </div>
                    </div>
                  );
                })
              )}

            </div>

            <div className="mt-4 p-3 rounded-lg bg-muted/50 border text-sm flex gap-2 leading-relaxed">
              <Flame className="h-4 w-4 text-red-brand shrink-0 mt-0.5" />
              <span>{displayScenario.reason}</span>
            </div>
          </Card>

          <Button
            size="lg"
            onClick={send}
            disabled={displayScenario.items.length === 0 || (v22Loading && !v22Result)}
            className="w-full h-14 text-base font-semibold text-white shadow-lg active:scale-[0.99] transition"
            style={{ backgroundColor: "#008060" }}
          >
            <Send className="h-5 w-5 mr-2" />
            Send to Tablet
          </Button>
        </div>
      </div>
    </div>
  );
}

function formatHour(h: number): string {
  const hr = ((h % 12) + 12) % 12 || 12;
  const ap = h >= 12 ? "PM" : "AM";
  return `${hr}:00 ${ap}`;
}

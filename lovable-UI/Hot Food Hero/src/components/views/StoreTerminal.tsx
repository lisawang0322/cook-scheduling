import { useState, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Bell,
  Clock,
  CheckCircle2,
  Check,
  Minus,
  Plus,
  Flame,
  ChevronRight,
  PartyPopper,
  Sparkles,
} from "lucide-react";
import { useScenario } from "@/lib/scenario-context";
import { addMinutes, addHours, type CookItem, type Scenario } from "@/lib/hot-food";
import { logAction } from "@/lib/api/scheduler.functions";

const PRIORITY = [
  { label: "COOK NOW", chip: "bg-red-brand text-white" },
  { label: "NEXT", chip: "bg-yellow-400 text-black" },
  { label: "THEN", chip: "bg-primary text-primary-foreground" },
];

type Step = 1 | 2 | 3 | 4;

export function StoreTerminal() {
  const { activeScenario: scenario, notificationNonce } = useScenario();

  const [step, setStep] = useState<Step>(1);
  const [notifVisible, setNotifVisible] = useState(false);
  const [chime, setChime] = useState(false);

  const [quantities, setQuantities] = useState<Record<string, number>>({});
  const [confirmed, setConfirmed] = useState<Record<string, boolean>>({});

  // Reset quantities + jump to standby when scenario changes
  useEffect(() => {
    const q: Record<string, number> = {};
    scenario.items.forEach((i) => (q[i.id] = i.recommended));
    setQuantities(q);
    setConfirmed({});
    setStep(1);
  }, [scenario]);

  // Fire notification when entering step 1 or when an external trigger bumps the nonce
  useEffect(() => {
    if (step !== 1) return;
    setNotifVisible(false);
    setChime(false);
    const t = setTimeout(() => {
      setNotifVisible(true);
      setChime(true);
    }, 500);
    return () => clearTimeout(t);
  }, [step, notificationNonce, scenario.id]);

  const snooze = () => {
    setNotifVisible(false);
    setChime(false);
    setTimeout(() => {
      setNotifVisible(true);
      setChime(true);
    }, 4000);
  };

  const openCookList = () => {
    setNotifVisible(false);
    setStep(2);
  };

  const adjustQty = (item: CookItem, delta: number) => {
    setQuantities((prev) => {
      const next = Math.max(0, (prev[item.id] ?? 0) + delta * item.lcu);
      return { ...prev, [item.id]: next };
    });
  };

  const confirmItem = async (item: CookItem) => {
    setConfirmed((prev) => ({ ...prev, [item.id]: true }));
    const record = {
      ts: new Date().toISOString(),
      scenario: scenario.id,
      store: scenario.storeId,
      item: item.id,
      recommended_qty: item.recommended,
      confirmed_qty: quantities[item.id],
      delta: quantities[item.id] - item.recommended,
      lcu: item.lcu,
    };
    logAction({ data: record }).catch((err) =>
      console.error("[HotFoodAssistant] log-action failed", err)
    );
  };

  const allConfirmed = useMemo(
    () => scenario.items.length > 0 && scenario.items.every((i) => confirmed[i.id]),
    [confirmed, scenario]
  );

  const startCook = () => setStep(4);
  const nextBatch = () => setStep(1);

  return (
    <div className="relative mx-auto w-full max-w-md min-h-[760px] bg-white rounded-3xl border shadow-xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-2 text-[11px] font-medium text-muted-foreground bg-gray-50 border-b">
        <span>{scenario.time}</span>
        <span className="flex items-center gap-1">
          <Sparkles className="h-3 w-3 text-primary" /> Hot Food Assistant
        </span>
        <span>Store {scenario.storeId}</span>
      </div>

      {step === 1 && (
        <LockScreen
          scenario={scenario}
          notifVisible={notifVisible}
          chime={chime}
          onOpen={openCookList}
          onSnooze={snooze}
        />
      )}
      {step === 2 && (
        <CookList scenario={scenario} onConfirm={() => setStep(3)} onBack={() => setStep(1)} />
      )}
      {step === 3 && (
        <ConfirmAmounts
          scenario={scenario}
          quantities={quantities}
          confirmed={confirmed}
          adjustQty={adjustQty}
          confirmItem={confirmItem}
          allConfirmed={allConfirmed}
          onStart={startCook}
          onBack={() => setStep(2)}
        />
      )}
      {step === 4 && <Success onNext={nextBatch} />}
    </div>
  );
}

function LockScreen({
  scenario,
  notifVisible,
  chime,
  onOpen,
  onSnooze,
}: {
  scenario: Scenario;
  notifVisible: boolean;
  chime: boolean;
  onOpen: () => void;
  onSnooze: () => void;
}) {
  return (
    <div className="relative h-[720px] bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white flex flex-col items-center justify-center px-6">
      <div className="text-xs uppercase tracking-[0.3em] opacity-60">7-Eleven</div>
      <div className="mt-2 text-base opacity-80">Store {scenario.storeId} · {scenario.storeType}</div>
      <div className="mt-8 text-7xl font-bold tabular-nums">{scenario.time}</div>
      <div className="mt-2 text-sm opacity-70">{scenario.dayLabel}</div>

      <div className="mt-10 flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/15 text-sm">
        <CheckCircle2 className="h-4 w-4 text-emerald-400" />
        All caught up
      </div>

      <div className="absolute bottom-8 inset-x-6 text-center text-xs opacity-50">
        Waiting for next cook alert…
      </div>

      <div
        className={`absolute top-4 inset-x-4 transition-all duration-500 ${
          notifVisible
            ? "translate-y-0 opacity-100"
            : "-translate-y-20 opacity-0 pointer-events-none"
        }`}
      >
        <div className="bg-white text-foreground rounded-2xl shadow-2xl border border-black/5 overflow-hidden">
          <div className="flex items-start gap-3 p-4">
            <div className={`h-10 w-10 rounded-xl bg-primary text-white flex items-center justify-center shrink-0 ${chime ? "animate-bounce" : ""}`}>
              <Bell className="h-5 w-5" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <div className="font-semibold text-sm">Time to Cook 🍕</div>
                <div className="text-[10px] text-muted-foreground">now</div>
              </div>
              <div className="text-sm text-muted-foreground mt-0.5">
                The oven is free and items are due. Tap to see what to cook.
              </div>
              <div className="text-[11px] text-muted-foreground mt-1">
                Store {scenario.storeId} · {scenario.dayLabel} {scenario.time}
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 border-t">
            <button
              onClick={onSnooze}
              className="py-3 text-sm font-medium text-muted-foreground hover:bg-muted transition active:scale-[0.98]"
            >
              Snooze 5 min
            </button>
            <button
              onClick={onOpen}
              className="py-3 text-sm font-semibold text-primary border-l hover:bg-primary/5 transition active:scale-[0.98] flex items-center justify-center gap-1"
            >
              View Cook List <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function CookList({
  scenario,
  onConfirm,
}: {
  scenario: Scenario;
  onConfirm: () => void;
  onBack: () => void;
}) {
  return (
    <div className="flex flex-col h-[720px]">
      <div className="px-5 pt-5 pb-3">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          {scenario.dayLabel} · {scenario.time} · {scenario.storeType}
        </div>
        <h1 className="text-2xl font-bold mt-1">Cook in this order</h1>
      </div>

      <div className="flex-1 overflow-y-auto px-5 pb-4 space-y-3">
        {scenario.items.length === 0 && (
          <div className="text-sm text-muted-foreground p-6 text-center border rounded-xl">
            No items in this scenario. Use the Scenario Simulator to send one.
          </div>
        )}
        {scenario.items.map((item, idx) => {
          const p = PRIORITY[Math.min(idx, 2)];
          const ready = addMinutes(scenario.time, item.cookTimeMin);
          const expiry = addHours(ready, item.holdTimeHr);
          return (
            <Card
              key={item.id}
              className={`p-4 border-l-4 ${idx === 0 ? "border-l-red-brand" : idx === 1 ? "border-l-yellow-400" : "border-l-primary"} transition active:scale-[0.99]`}
            >
              <div className="flex items-start gap-3">
                <div className="text-4xl leading-none">{item.emoji}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge className={`${p.chip} px-2 py-0.5 text-[10px] font-bold tracking-wider`}>
                      {p.label}
                    </Badge>
                    <h3 className="font-bold text-lg leading-tight">{item.name}</h3>
                  </div>
                  <div className="mt-2 text-xl font-semibold">
                    Cook {item.recommended} units
                    <span className="ml-1 text-sm font-normal text-muted-foreground">
                      ({Math.max(1, Math.round(item.recommended / item.lcu))} {item.lcu > 1 ? `batches of ${item.lcu}` : "tray"})
                    </span>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                    <span className="inline-flex items-center gap-1"><Clock className="h-3 w-3" /> {item.cookTimeMin} min</span>
                    <span>Ready {ready}</span>
                    <span>Discard {expiry}</span>
                  </div>
                </div>
              </div>
            </Card>
          );
        })}

        <div className="rounded-xl border-2 border-dashed border-orange/50 bg-orange/5 p-4">
          <div className="flex items-start gap-3">
            <div className="text-3xl">🌯</div>
            <div className="flex-1">
              <div className="text-xs font-semibold text-orange uppercase tracking-wider">
                Roller Grill · Parallel
              </div>
              <div className="font-semibold mt-0.5">Taquitos</div>
              <div className="text-xs text-muted-foreground mt-1">
                Fill to capacity — no scheduling needed.
              </div>
            </div>
          </div>
        </div>

        <div className="text-xs text-muted-foreground leading-relaxed flex gap-2 px-1">
          <Flame className="h-4 w-4 text-red-brand shrink-0 mt-0.5" />
          <span>{scenario.reason}</span>
        </div>
      </div>

      <div className="p-4 border-t bg-white">
        <Button
          size="lg"
          disabled={scenario.items.length === 0}
          className="w-full h-14 text-base font-semibold bg-primary hover:bg-primary/90 active:scale-[0.98] transition"
          onClick={onConfirm}
        >
          Confirm Quantities
          <ChevronRight className="ml-1 h-5 w-5" />
        </Button>
      </div>
    </div>
  );
}

function ConfirmAmounts({
  scenario,
  quantities,
  confirmed,
  adjustQty,
  confirmItem,
  allConfirmed,
  onStart,
}: {
  scenario: Scenario;
  quantities: Record<string, number>;
  confirmed: Record<string, boolean>;
  adjustQty: (item: CookItem, delta: number) => void;
  confirmItem: (item: CookItem) => void;
  allConfirmed: boolean;
  onStart: () => void;
  onBack: () => void;
}) {
  return (
    <div className="flex flex-col h-[720px]">
      <div className="px-5 pt-5 pb-3">
        <h1 className="text-2xl font-bold">Confirm how much you're cooking</h1>
        <div className="text-xs text-muted-foreground mt-1">
          Tap − / + to adjust. Quantities snap to batch size.
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 pb-4 space-y-3">
        {scenario.items.map((item, idx) => {
          const qty = quantities[item.id] ?? item.recommended;
          const isConfirmed = !!confirmed[item.id];
          const delta = qty - item.recommended;
          return (
            <Card
              key={item.id}
              className={`p-4 transition ${isConfirmed ? "bg-emerald-50 border-emerald-300" : ""}`}
            >
              <div className="flex items-center gap-3">
                <div className="text-3xl">{item.emoji}</div>
                <div className="flex-1 min-w-0">
                  <div className="font-semibold leading-tight">{item.name}</div>
                  <div className="text-[11px] text-muted-foreground">
                    Recommended {item.recommended} · batch of {item.lcu}
                  </div>
                </div>
                {isConfirmed && (
                  <div className="h-9 w-9 rounded-full bg-emerald-500 text-white flex items-center justify-center shadow">
                    <Check className="h-5 w-5" />
                  </div>
                )}
              </div>

              <div className="mt-4 flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 bg-muted rounded-2xl p-1">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-12 w-12 rounded-xl active:scale-90 transition"
                    onClick={() => adjustQty(item, -1)}
                    disabled={qty <= 0 || isConfirmed}
                    aria-label={`Decrease ${item.name}`}
                  >
                    <Minus className="h-5 w-5" />
                  </Button>
                  <div className="min-w-[64px] text-center">
                    <div className="text-3xl font-bold tabular-nums leading-none">{qty}</div>
                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground mt-0.5">
                      units
                    </div>
                  </div>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-12 w-12 rounded-xl active:scale-90 transition"
                    onClick={() => adjustQty(item, 1)}
                    disabled={isConfirmed}
                    aria-label={`Increase ${item.name}`}
                  >
                    <Plus className="h-5 w-5" />
                  </Button>
                </div>

                {!isConfirmed ? (
                  <Button
                    onClick={() => confirmItem(item)}
                    className="h-12 px-5 bg-primary hover:bg-primary/90 active:scale-[0.97] transition font-semibold"
                  >
                    Confirm
                  </Button>
                ) : (
                  <div className="text-xs text-emerald-700 font-medium text-right">
                    Locked in
                    {delta !== 0 && (
                      <div className="text-[10px] text-muted-foreground">
                        {delta > 0 ? `+${delta}` : delta} vs rec
                      </div>
                    )}
                  </div>
                )}
              </div>
              {idx === 0 && !isConfirmed && (
                <div className="mt-2 text-[10px] uppercase tracking-wider text-red-brand font-semibold">
                  Cook now
                </div>
              )}
            </Card>
          );
        })}
      </div>

      <div className="p-4 border-t bg-white">
        <Button
          size="lg"
          disabled={!allConfirmed}
          onClick={onStart}
          className="w-full h-14 text-base font-semibold bg-primary hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground active:scale-[0.98] transition"
        >
          {allConfirmed ? "Start Cook" : `Confirm all ${scenario.items.length} items`}
        </Button>
      </div>
    </div>
  );
}

function Success({ onNext }: { onNext: () => void }) {
  return (
    <div className="h-[720px] flex flex-col items-center justify-center text-center px-6 bg-gradient-to-b from-emerald-50 to-white">
      <div className="h-24 w-24 rounded-full bg-primary text-white flex items-center justify-center shadow-xl animate-[scale-in_0.3s_ease-out]">
        <PartyPopper className="h-12 w-12" />
      </div>
      <h2 className="mt-6 text-2xl font-bold">Cooking started</h2>
      <p className="mt-2 text-muted-foreground max-w-xs">
        Load the trays and press the oven program. We'll ping you for the next batch.
      </p>
      <Button
        size="lg"
        onClick={onNext}
        className="mt-10 h-14 px-8 bg-primary hover:bg-primary/90 active:scale-[0.98] transition font-semibold"
      >
        Next Batch
        <ChevronRight className="ml-1 h-5 w-5" />
      </Button>
    </div>
  );
}

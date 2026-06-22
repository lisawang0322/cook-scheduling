import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Monitor, BarChart3, Sliders, TrendingUp, HelpCircle, Flame, Wand2, Tablet } from "lucide-react";
import { Toaster } from "@/components/ui/sonner";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { StoreTerminal } from "@/components/views/StoreTerminal";
import { ScenarioComparison } from "@/components/views/ScenarioComparison";
import { WhatIfSimulator } from "@/components/views/WhatIfSimulator";
import { ImpactDashboard } from "@/components/views/ImpactDashboard";
import { ScenarioSimulator } from "@/components/views/ScenarioSimulator";
import { ScenarioProvider } from "@/lib/scenario-context";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "7-Eleven Hot Food Cook Scheduling Optimizer" },
      { name: "description", content: "ML-powered cook scheduling for 7-Eleven hot food across overlapping daypart windows." },
    ],
  }),
  component: App,
});

const NAV = [
  { id: "assistant", label: "Hot Food Assistant", sub: "Simulator + Tablet", Icon: Monitor, active: true },
  { id: "compare", label: "Scenario Comparison", sub: "Coming soon", Icon: BarChart3, active: false },
  { id: "whatif", label: "What-If Simulator", sub: "Coming soon", Icon: Sliders, active: false },
  { id: "impact", label: "Impact Dashboard", sub: "Coming soon", Icon: TrendingUp, active: false },
] as const;

type ViewId = "assistant";
type AssistantMode = "simulator" | "tablet";

function App() {
  const [view, setView] = useState<ViewId>("assistant");
  const [mode, setMode] = useState<AssistantMode>("simulator");

  return (
    <ScenarioProvider>
      <TooltipProvider delayDuration={150}>
        <div className="min-h-screen flex bg-background">
          <aside className="hidden md:flex w-72 shrink-0 flex-col bg-sidebar text-sidebar-foreground">
            <div className="p-6 border-b border-sidebar-border">
              <div className="flex items-center gap-3">
                <div className="h-11 w-11 rounded-xl orange-gradient flex items-center justify-center shadow-lg">
                  <Flame className="h-6 w-6 text-white" />
                </div>
                <div>
                  <div className="font-bold text-base leading-tight">7-Eleven</div>
                  <div className="text-xs opacity-75">Cook Optimizer</div>
                </div>
              </div>
            </div>
            <nav className="flex-1 p-3 space-y-1">
              {NAV.map((n) => {
                const isActive = n.active && view === n.id;
                return (
                  <button
                    key={n.id}
                    onClick={() => n.active && setView(n.id as ViewId)}
                    disabled={!n.active}
                    className={`w-full text-left flex items-start gap-3 px-3 py-3 rounded-lg transition-all ${
                      isActive
                        ? "bg-sidebar-primary text-sidebar-primary-foreground shadow-md scale-[1.01]"
                        : n.active
                        ? "hover:bg-sidebar-accent"
                        : "opacity-40 cursor-not-allowed"
                    }`}
                  >
                    <n.Icon className="h-5 w-5 mt-0.5 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-sm">{n.label}</div>
                      <div className="text-xs opacity-75">{n.sub}</div>
                    </div>
                  </button>
                );
              })}
            </nav>
            <div className="p-4 border-t border-sidebar-border text-xs opacity-70 leading-relaxed">
              v2.2 · ML Optimizer<br />Pairwise + temporal aggregation
            </div>
          </aside>

          <div className="md:hidden fixed top-0 inset-x-0 z-40 bg-sidebar text-sidebar-foreground flex overflow-x-auto">
            {NAV.map((n) => (
              <button
                key={n.id}
                onClick={() => n.active && setView(n.id as ViewId)}
                disabled={!n.active}
                className={`flex-1 min-w-[110px] py-3 px-2 text-xs flex flex-col items-center gap-1 ${
                  n.active && view === n.id ? "bg-sidebar-primary text-sidebar-primary-foreground" : ""
                } ${!n.active ? "opacity-40 cursor-not-allowed" : ""}`}
              >
                <n.Icon className="h-4 w-4" />
                {n.label.split(" ")[0]}
              </button>
            ))}
          </div>

          <main className="flex-1 min-w-0 p-6 md:p-8 pt-20 md:pt-8 max-w-[1400px] mx-auto w-full">
            {view === "assistant" && (
              <div className="space-y-6">
                <div className="inline-flex p-1 rounded-xl bg-muted border">
                  <button
                    onClick={() => setMode("simulator")}
                    className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition ${
                      mode === "simulator"
                        ? "bg-white shadow text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <Wand2 className="h-4 w-4" /> Scenario Simulator
                  </button>
                  <button
                    onClick={() => setMode("tablet")}
                    className={`px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition ${
                      mode === "tablet"
                        ? "bg-white shadow text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <Tablet className="h-4 w-4" /> Associate Tablet
                  </button>
                </div>
                {mode === "simulator" ? (
                  <ScenarioSimulator onSendToTablet={() => setMode("tablet")} />
                ) : (
                  <StoreTerminal />
                )}
              </div>
            )}
          </main>

          <div className="fixed bottom-5 right-5 z-50">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button size="icon" className="rounded-full h-12 w-12 shadow-xl bg-primary hover:bg-primary/90">
                  <HelpCircle className="h-5 w-5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left" className="max-w-xs text-xs leading-relaxed">
                <p className="font-semibold mb-1">About this prototype</p>
                <p className="opacity-90">Demo seeded with synthetic store-level data. Predictions blend a v2.2 pairwise ML model with hour-of-day temporal aggregation; falls back to the v1 rules-based scheduler if signals are missing.</p>
              </TooltipContent>
            </Tooltip>
          </div>

          <Toaster richColors position="top-right" />
        </div>
      </TooltipProvider>
    </ScenarioProvider>
  );
}

import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import { DEFAULT_SCENARIO, type Scenario } from "./hot-food";

type Ctx = {
  activeScenario: Scenario;
  setActiveScenario: (s: Scenario) => void;
  notificationNonce: number;
  triggerNotification: () => void;
};

const ScenarioContext = createContext<Ctx | null>(null);

export function ScenarioProvider({ children }: { children: ReactNode }) {
  const [activeScenario, setScenario] = useState<Scenario>(DEFAULT_SCENARIO);
  const [notificationNonce, setNonce] = useState(0);

  const setActiveScenario = useCallback((s: Scenario) => setScenario(s), []);
  const triggerNotification = useCallback(() => setNonce((n) => n + 1), []);

  return (
    <ScenarioContext.Provider
      value={{ activeScenario, setActiveScenario, notificationNonce, triggerNotification }}
    >
      {children}
    </ScenarioContext.Provider>
  );
}

export function useScenario() {
  const ctx = useContext(ScenarioContext);
  if (!ctx) throw new Error("useScenario must be used within ScenarioProvider");
  return ctx;
}

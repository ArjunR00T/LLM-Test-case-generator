import React, { createContext, useContext, useState } from 'react';

const TaskContext = createContext<{ taskId: string | null; setTaskId: React.Dispatch<React.SetStateAction<string | null>> } | undefined>(undefined);

export const TaskProvider = ({ children }: { children: React.ReactNode }) => {
  const [taskId, setTaskId] = useState<string | null>(null);
  return (
    <TaskContext.Provider value={{ taskId, setTaskId }}>
      {children} {/* this is required to let your app render */}
    </TaskContext.Provider>
  );
};

export const useTask = () => useContext(TaskContext);

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import TestCaseGenerator from './components/TestCaseGenerator';
import FileUpload from './components/FileUpload';
import Dashboard from './components/Dashboard';
import { TaskProvider } from './components/TaskContext';

function App() {
  return (
    <TaskProvider>
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/generator" element={<TestCaseGenerator />} />
          <Route path="/bulk-upload" element={<FileUpload />} />
        </Routes>
      </div>
    </Router>
    </TaskProvider>
  );
}

export default App;
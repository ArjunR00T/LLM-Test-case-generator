import React, { useEffect, useState } from 'react';
// import { Loader2, CheckCircle, ChevronDown, ChevronUp, TestTube2, Copy, Download, Save } from 'lucide-react';
import { Loader2, CheckCircle, TestTube2 } from 'lucide-react';
import axios from 'axios';
import { TaskProvider, useTask } from './TaskContext';

interface TestCase {
  id: string;
  title: string;
  steps: string[];
  expectedResult: string;
  priority: 'Critical' | 'High' | 'Medium' | 'Low';
  category: string;
}

let cnt:number = 0
const BaseUrl = "https://port-investment-allocation-organisations.trycloudflare.com";

interface SimilarExample {
  user_story: string;
  test_cases: TestCase[];
}


const TestCaseGenerator: React.FC = () => {
  const taskContext = useTask();


  const [userStory, setUserStory] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [generatedTestCases, setGeneratedTestCases] = useState<TestCase[]>([]);
  const [similarExamples, setSimilarExamples] = useState<SimilarExample[]>([]);
  // const [showSimilarExamples, setShowSimilarExamples] = useState(false);
  const [hasGenerated, setHasGenerated] = useState(false);
  const [activeTab, setActiveTab] = useState("testCases");
  
  const pollTaskStatus = (
    taskId: string,
    onSuccess: (data: any) => void,
    onError: (error: any) => void,
    interval: number = 6000
  ): NodeJS.Timeout => {
    const intervalId = setInterval(async () => {
      console.log("Count:"+cnt)
  
      
      try {
        if (taskId === null || taskId === undefined) {
          return;
        }
        const statusResponse = await axios.get(`${BaseUrl}/result/${taskId}`);
      
        const taskStatus = statusResponse.data;
        console.log(`Task ${taskId} Status: ${taskStatus.status}, Result: ${taskStatus.result}%`);
        // cnt += 1;
        // if (cnt == 5){
        //   taskStatus.status = "done"
        // }
        if (taskStatus.status === 'done') {
          clearInterval(intervalId);
          onSuccess(taskStatus.result);
        } else if (taskStatus.status === 'failed') {
          clearInterval(intervalId);
          onError(taskStatus.result);
        }
        
        // If pending/processing, do nothing
      } catch (error) {
        clearInterval(intervalId);
        onError(error);
      }
    }, interval);
  
    return intervalId; // So the caller can cancel it if needed
  };
  
  useEffect(() => {
    console.log("Starting useEffect fetch");
    const fetchStatus = async () => {
      const jobId = taskContext?.taskId;
      if (!jobId) return;
      
       const statusResponse = await axios.get(`${BaseUrl}/result/${jobId}`);
    
      const taskStatus = statusResponse.data;
      console.log("Inital fetched status: "+taskStatus.status);
  
      if (taskStatus.status === 'done') {
        setGeneratedTestCases(taskStatus.result[0] || []);
        setSimilarExamples(taskStatus.result[1] || []);
        setHasGenerated(true);
        setIsLoading(false);
      } else if (taskStatus.status === 'processing') {
        setHasGenerated(false);
        setIsLoading(true);
        pollTaskStatus(
          taskStatus.id,
          (resultData) => {
            setGeneratedTestCases(resultData[0] || []);
            setSimilarExamples(resultData[1] || []);
            setHasGenerated(true);
            setIsLoading(false);
            console.log('Polling complete. Data:', resultData);
          },
          (error) => {
            console.error("Polling error/failure:", error);
            alert(`Task failed: ${typeof error === 'string' ? error : JSON.stringify(error)}`);
            setHasGenerated(false);
            setIsLoading(false);
          }
        );
      } else {
        console.log(`Task ${jobId} status not recognized`);
      }
    };
  
    fetchStatus();
  }, [taskContext?.taskId]);
  

  const generateTestCases = async () => {
    // 1. Initial Loading State for the whole process
    setIsLoading(true);
    setHasGenerated(false); // Reset generated state
    setGeneratedTestCases([]); // Clear previous results
    setSimilarExamples([]); // Clear previous similar examples

    let data:any = { similiar: [], test_cases: [] };
    let taskId:any = null;
    let pollingIntervalId:any = null; // To store the interval ID for clearing

    try {
        console.log("Starting test case generation for user story...");

        // --- Step 1: Request to start the background task ---
        const startResponse = await axios.post(
            `${BaseUrl}/generate`,
            { inp_user_story: userStory }
        );

        // const startResponse = {
        //   data: {
        //     id: "taskId",
        //     message: "Task started successfully"
        //   }
        // }

        taskId = startResponse.data.id;
         // Extract the task_id
        
        if (taskContext) {
            taskContext.setTaskId(taskId); // Store taskId in context
        }

        const message = startResponse.data.message;
        console.log(message);
        console.log("Task ID received:", taskId);

        if (!taskId) {
            throw new Error("Task ID not received from backend.");
        }

        // --- Step 2: Start Polling for Task Status ---
        // You might want to set a different loading state here, e.g., 'isPolling'
        // or just rely on the main isLoading to show that *something* is happening.

        // Set up the polling interval
        // Poll every 5 seconds (adjust as needed, typically 2-10 seconds is fine)
        pollTaskStatus(
          taskId,
          (resultData) => {
            setGeneratedTestCases(resultData[0] || []);
            setSimilarExamples(resultData[1] || []);
            setHasGenerated(true);
            setIsLoading(false);
            console.log('Polling complete. Data:', resultData);
          },
          (error) => {
            console.error("Polling error/failure:", error);
            alert(`Task failed: ${typeof error === 'string' ? error : JSON.stringify(error)}`);
            setHasGenerated(false);
            setIsLoading(false);
          }
        );

    } catch (error: any) {
        // Handle errors from the initial POST request
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId); // Ensure any pending polling is cleared
        }
        if (error.response) {
            console.error("Initial Server Error:", error.response.status, error.response.data);
            alert(`API error ${error.response.status}: ${JSON.stringify(error.response.data)}`);
        } else if (error.request) {
            console.error("No response received for initial request:", error.request);
            alert("No response from API");
        } else {
            console.error("Error setting up initial request:", error.message);
            alert("Unexpected error: " + error.message);
        }
        setIsLoading(false); // Stop loading indicator on initial failure
        setHasGenerated(false); // Keep hasGenerated false on initial failure
    }
    // Note: setIsLoading(false) is now moved into the polling success/failure blocks
    // because the main request completes quickly, but the overall process is still "loading"
    // until the polling finishes.
};
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };


  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center space-x-3 mb-2">
          <TestTube2 className="h-8 w-8 text-blue-600" />
          <h1 className="text-3xl font-bold text-gray-900">Test Case Generator</h1>
        </div>
        <p className="text-gray-600">Generate comprehensive test cases from user stories using AI-powered analysis</p>
      </div>



      {/* Main Input Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
        <label htmlFor="userStory" className="block text-sm font-semibold text-gray-700 mb-3">
          User Story Input
        </label>
        <textarea
          id="userStory"
          value={userStory}
          onChange={(e) => setUserStory(e.target.value)}
          className="w-full h-32 p-4 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-900 placeholder-gray-500 bg-white"
          placeholder="Input user story...."
        />

        <div className="flex items-center justify-between mt-4">
          <div className="text-sm text-gray-500">
            {userStory.length}/500 characters
          </div>
          <button
            onClick={generateTestCases}
            disabled={!userStory.trim() || isLoading}
            className="bg-blue-600 text-white font-semibold py-2 px-6 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 flex items-center space-x-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Generating...</span>
              </>
            ) : (
              <>
                <TestTube2 className="h-4 w-4" />
                <span>Generate Test Cases</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Generated Test Cases */}
      {(isLoading || hasGenerated) && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900 flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              <span>Generated Test Cases</span>
            </h2>
            {/* {hasGenerated && (
              <div className="flex items-center space-x-2">
                <button className="flex items-center space-x-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-900 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors duration-200">
                  <Download className="h-4 w-4" />
                  <span>Export</span>
                </button>
                <button className="flex items-center space-x-1 px-3 py-1 text-sm text-white bg-blue-600 hover:bg-blue-700 rounded-md transition-colors duration-200">
                  <Save className="h-4 w-4" />
                  <span>Save to Project</span>
                </button>
              </div>
            )} */}
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="flex flex-col items-center space-y-4">
                <Loader2 className="h-12 w-12 animate-spin text-blue-600" />
                <p className="text-gray-600">AI is analyzing your user story and generating comprehensive test cases...</p>
                <div className="w-64 bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex space-x-4 border-b mb-6">
              <button
                className={`pb-2 px-4 font-medium ${activeTab === "testCases" ? "border-b-2 border-blue-600 text-blue-600" : "text-gray-500 hover:text-blue-600"}`}
                onClick={() => setActiveTab("testCases")}
              >
                Generated Test Cases
              </button>
              <button
                className={`pb-2 px-4 font-medium ${activeTab === "references" ? "border-b-2 border-blue-600 text-blue-600" : "text-gray-500 hover:text-blue-600"}`}
                onClick={() => setActiveTab("references")}
              >
                Referenced From
              </button>
            </div>

          )}

          {activeTab === "testCases" && (
            <div className="space-y-4">
              {generatedTestCases.map((testCase) => (
                <div key={testCase.id} className="bg-white p-4 rounded-md border border-gray-200">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-gray-900">{testCase.title}</h4>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getPriorityColor(testCase.priority)}`}>
                        {testCase.priority.charAt(0).toUpperCase() + testCase.priority.slice(1)}
                      </span>
                      <span className="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-700 border border-gray-200">
                        {testCase.category.charAt(0).toUpperCase() + testCase.category.slice(1)}
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-gray-600">{testCase.expectedResult}</p>
                </div>
              ))}
            </div>
          )}

          {activeTab === "references" && (
            <div className="space-y-4">
              {similarExamples.map((example, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-5 bg-gray-50">
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-semibold text-gray-700">User Story:</h3>
                    </div>
                    <p className="text-sm text-gray-600 italic bg-blue-50 p-3 rounded-md border border-blue-200">
                      "{example.user_story}"
                    </p>
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold text-gray-700 mb-3">Test Cases:</h3>
                    <div className="space-y-3">
                      {example.test_cases.map((text, index) => {
                        const testCase = {
                          id: index,
                          title: `Test Case ${index + 1}`,
                          expectedResult: text
                        };

                        return (
                          <div key={testCase.id} className="bg-white p-4 rounded-md border border-gray-200">
                            <div className="flex items-center justify-between mb-2">
                              <h4 className="text-sm font-medium text-gray-900">{testCase.title}</h4>
                            </div>
                            <p className="text-xs text-gray-600">{testCase.expectedResult}</p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

        </div>
      )}


      {/* <style jsx>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
      `}</style> */}
    </div>
  );
};

export default TestCaseGenerator;

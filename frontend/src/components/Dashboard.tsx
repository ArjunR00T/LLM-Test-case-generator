import React from 'react';
import { TestTube2, FileText, Clock, CheckCircle, AlertTriangle } from 'lucide-react';
// import { TestTube2, Upload, FileText, TrendingUp, Clock, CheckCircle, AlertTriangle } from 'lucide-react';

// import { Link } from 'react-router-dom';

const Dashboard: React.FC = () => {
  const stats = [
    {
      title: 'Test Cases Generated',
      value: '2,847',
      changeType: 'positive',
      icon: TestTube2,
      color: 'blue'
    },
    {
      title: 'Stories Processed',
      value: '156',
      changeType: 'positive',
      icon: FileText,
      color: 'green'
    }
  ];

  const recentActivity = [
    {
      id: 1,
      action: 'Generated test cases for user authentication flow',
      user: 'Sarah Chen',
      time: '2 minutes ago',
      status: 'completed',
      testCases: 8
    },
    {
      id: 2,
      action: 'Bulk upload processed - Payment module stories',
      user: 'Mike Johnson',
      time: '15 minutes ago',
      status: 'completed',
      testCases: 24
    },
    {
      id: 3,
      action: 'Generated test cases for shopping cart functionality',
      user: 'Emily Rodriguez',
      time: '1 hour ago',
      status: 'completed',
      testCases: 12
    },
    {
      id: 4,
      action: 'Bulk upload processing - User management stories',
      user: 'David Kim',
      time: '2 hours ago',
      status: 'processing',
      testCases: 0
    }
  ];

  const getStatColor = (color: string) => {
    const colors = {
      blue: 'bg-blue-50 text-blue-600',
      green: 'bg-green-50 text-green-600',
      purple: 'bg-purple-50 text-purple-600',
      orange: 'bg-orange-50 text-orange-600'
    };
    return colors[color as keyof typeof colors] || colors.blue;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Clock className="h-4 w-4 text-orange-500" />;
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Quality Assurance Dashboard</h1>
        <p className="text-gray-600 mt-1">Monitor test case generation and team productivity</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                <p className="text-2xl font-bold text-gray-900 mt-2">{stat.value}</p>
                <div className="flex items-center mt-2">
                  <span className={`text-sm font-medium ${stat.changeType === 'positive' ? 'text-green-600' : 'text-red-600'
                    }`}>
                  </span>
                </div>
              </div>
              <div className={`p-3 rounded-lg ${getStatColor(stat.color)}`}>
                <stat.icon className="h-6 w-6" />
              </div>
            </div>
          </div>
        ))}
      </div>


      <div className="">
        {/* Recent Activity */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">Recent Activity</h2>
              <button className="text-sm text-blue-600 hover:text-blue-700 font-medium">
                View All
              </button>
            </div>

            <div className="space-y-4">
              {recentActivity.map((activity) => (
                <div key={activity.id} className="flex items-start space-x-3 p-3 rounded-lg hover:bg-gray-50 transition-colors duration-200">
                  <div className="flex-shrink-0 mt-1">
                    {getStatusIcon(activity.status)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">{activity.action}</p>
                    <div className="flex items-center space-x-4 mt-1">
                      <p className="text-sm text-gray-500">by {activity.user}</p>
                      <p className="text-sm text-gray-500">{activity.time}</p>
                      {activity.testCases > 0 && (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          {activity.testCases} test cases
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
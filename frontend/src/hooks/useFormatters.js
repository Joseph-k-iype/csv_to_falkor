import { useMemo } from 'react';
import { TrendingUp, Target, Lightbulb, Shield, Brain } from 'lucide-react';

export const useFormatters = () => {
  const formatAdvancedInsight = useMemo(() => (text) => {
    const sections = text.split('\n\n');
    const formattedSections = [];
    
    sections.forEach((section, index) => {
      const lowerSection = section.toLowerCase();
      
      if (lowerSection.includes('summary') || lowerSection.includes('key findings') || lowerSection.includes('overview')) {
        formattedSections.push(
          <div key={index} className="mb-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
              <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
              Executive Summary
            </h3>
            <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg">
              <p className="text-gray-700 leading-relaxed">
                {section.replace(/^(Summary|Key Findings|Overview):\s*/i, '')}
              </p>
            </div>
          </div>
        );
      } else if (lowerSection.includes('strategic') || lowerSection.includes('business impact') || lowerSection.includes('implications')) {
        formattedSections.push(
          <div key={index} className="mb-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
              <Target className="w-5 h-5 mr-2 text-purple-600" />
              Strategic Analysis
            </h3>
            <div className="bg-purple-50 border-l-4 border-purple-400 p-4 rounded-r-lg">
              <p className="text-gray-700 leading-relaxed">{section}</p>
            </div>
          </div>
        );
      } else if (lowerSection.includes('recommendation') || lowerSection.includes('action') || lowerSection.includes('next steps')) {
        formattedSections.push(
          <div key={index} className="mb-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
              <Lightbulb className="w-5 h-5 mr-2 text-amber-600" />
              Strategic Recommendations
            </h3>
            <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded-r-lg">
              <p className="text-gray-700 leading-relaxed">{section}</p>
            </div>
          </div>
        );
      } else if (lowerSection.includes('risk') || lowerSection.includes('warning') || lowerSection.includes('concern')) {
        formattedSections.push(
          <div key={index} className="mb-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
              <Shield className="w-5 h-5 mr-2 text-red-600" />
              Risk Assessment
            </h3>
            <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-r-lg">
              <p className="text-gray-700 leading-relaxed">{section}</p>
            </div>
          </div>
        );
      } else if (lowerSection.includes('cypher') || lowerSection.includes('query') || lowerSection.includes('validation')) {
        formattedSections.push(
          <div key={index} className="mb-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
              <Brain className="w-5 h-5 mr-2 text-green-600" />
              Technical Analysis
            </h3>
            <div className="bg-green-50 border-l-4 border-green-400 p-4 rounded-r-lg">
              <p className="text-gray-700 leading-relaxed font-mono text-sm">{section}</p>
            </div>
          </div>
        );
      } else if (section.trim()) {
        formattedSections.push(
          <div key={index} className="mb-4">
            <p className="text-gray-700 leading-relaxed">{section}</p>
          </div>
        );
      }
    });
    
    return formattedSections;
  }, []);

  const categorizeQuestions = useMemo(() => (questions) => {
    const categories = {
      strategic: [],
      operational: [],
      technical: [],
      risk: []
    };
    
    questions.forEach(question => {
      const lowerQ = question.toLowerCase();
      if (lowerQ.includes('strategic') || lowerQ.includes('business') || lowerQ.includes('impact')) {
        categories.strategic.push(question);
      } else if (lowerQ.includes('operational') || lowerQ.includes('process') || lowerQ.includes('efficiency')) {
        categories.operational.push(question);
      } else if (lowerQ.includes('technical') || lowerQ.includes('cypher') || lowerQ.includes('query')) {
        categories.technical.push(question);
      } else if (lowerQ.includes('risk') || lowerQ.includes('security') || lowerQ.includes('threat')) {
        categories.risk.push(question);
      } else {
        categories.strategic.push(question);
      }
    });
    
    return categories;
  }, []);

  return {
    formatAdvancedInsight,
    categorizeQuestions
  };
};
"use client";

import React from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  ScatterChart, Scatter, ZAxis
} from 'recharts';
import {
  ArrowLeft, CheckCircle, Info, BarChart2, PieChart as PieIcon, Scale,
  LayoutGrid,
  BookOpen,
  Users,
  AlertTriangle, // <-- New Icon
  Zap // <-- New Icon
} from 'lucide-react';
// We use a regular <a> tag for navigation from the dashboard, so no 'next/link' needed.

// --- Data (No changes needed here) ---

// We'll simulate the SHAP plot data (based on typical results for this problem)
const featureImportanceData = [
  { name: 'AGE', importance: 0.28 },
  { name: 'WAIST_HT_RATIO', importance: 0.22 },
  { name: 'BMI', importance: 0.19 },
  { name: 'TG_TO_HDL', importance: 0.15 },
  { name: 'SBP_MEAN', importance: 0.11 },
  { name: 'PA_SCORE', importance: 0.05 },
  { name: 'Other', importance: 0.03 },
].sort((a, b) => b.importance - a.importance);

// Based on your classification report's 'support' (1960 vs 337)
const classDistributionData = [
  { name: 'Non-Diabetic', value: 1960, fill: '#3b82f6' }, // Blue
  { name: 'Diabetic', value: 337, fill: '#f472b6' }, // Pink
];

// --- More Realistic Scatterplot Data (250 points) ---
const generateData = (count, isDiabetic) => {
  const data = [];
  for (let i = 0; i < count; i++) {
    let age, bmi;
    const randAge = (Math.random() + Math.random()) / 2;
    const randBmi = (Math.random() + Math.random()) / 2;

    if (isDiabetic) {
      age = randAge * 40 + 35; // Skewed 35-75
      bmi = randBmi * 16 + 25; // Skewed 25-41
    } else {
      age = randAge * 40 + 20; // Skewed 20-60
      bmi = randBmi * 12 + 20; // Skewed 20-32
    }
    const outlierChance = Math.random();
    if (isDiabetic && outlierChance > 0.9) {
      bmi = Math.random() * 7 + 20; // BMI 20-27
    }
    if (!isDiabetic && outlierChance > 0.85) {
      bmi = Math.random() * 10 + 32; // BMI 32-42
    }

    data.push({ age: age, bmi: bmi, class: isDiabetic ? 1 : 0 });
  }
  return data;
}
const nonDiabeticData = generateData(200, false);
const diabeticData = generateData(50, true);
// --- End of data generation ---


// --- Re-usable Components ---
const MetricCard = ({ title, value, icon }) => (
  <motion.div
    className="bg-gray-900/50 p-4 rounded-lg border border-gray-700/50 flex items-center space-x-3"
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.3 }}
  >
    <div className="flex-shrink-0 p-2 bg-purple-500/20 rounded-full">
      {icon}
    </div>
    <div>
      <p className="text-sm text-gray-400">{title}</p>
      <p className="text-xl font-semibold text-gray-100">{value}</p>
    </div>
  </motion.div>
);

const ChartCard = ({ title, icon, children }) => (
  <motion.div
    className="bg-black/70 backdrop-blur-md rounded-lg shadow-2xl border border-gray-800/50 p-6"
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay: 0.2 }}
  >
    <h3 className="text-xl font-semibold mb-4 flex items-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
      {icon && React.cloneElement(icon, { className: 'mr-2' })}
      {title}
    </h3>
    <div className="w-full h-72">
      {children}
    </div>
  </motion.div>
);

// --- Main Dashboard Component ---
export default function DashboardPage() {
  return (
    <main className="flex flex-col items-center min-h-screen p-4 md:p-8 bg-black text-gray-200 overflow-y-auto relative font-sora">
      {/* --- CSS FIXES, FONT, & NEW SCROLLBAR STYLE --- */}
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');

        .font-sora {
          font-family: 'Sora', sans-serif;
        }

        /* Gradient Panning Animation */
        @keyframes gradient-pan {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient-pan {
          animation: gradient-pan 3s linear infinite;
        }

        /* --- Dark Scrollbar Styles --- */
        .dark-scrollbar::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        .dark-scrollbar::-webkit-scrollbar-track {
          background: #1f2937; /* gray-800 */
          border-radius: 10px;
        }
        .dark-scrollbar::-webkit-scrollbar-thumb {
          background-color: #4b5563; /* gray-600 */
          border-radius: 10px;
          border: 2px solid #1f2937; /* gray-800 */
        }
        .dark-scrollbar::-webkit-scrollbar-thumb:hover {
          background-color: #6b7280; /* gray-500 */
        }
        /* For Firefox */
        .dark-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: #4b5563 #1f2937;
        }
      `}</style>

      {/* --- Animated Background Glow --- */}
      <div className="fixed inset-0 z-0 overflow-hidden">
        <div className="absolute top-0 left-0 w-96 h-96 bg-purple-600/30 rounded-full blur-[150px] opacity-50 animate-pulse"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-pink-500/30 rounded-full blur-[150px] opacity-50 animate-pulse" style={{animationDelay: '3s'}}></div>
      </div>

      {/* --- Main Content (Scrollable) --- */}
      <div className="w-full max-w-5xl z-10 py-10">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Back Button */}
          <a
            href="/"
            className="flex items-center text-purple-400 hover:text-purple-300 transition-colors mb-6 group"
          >
            <ArrowLeft size={18} className="mr-2 transition-transform group-hover:-translate-x-1" />
            Back to Predictor
          </a>

          <h1 className="text-4xl md:text-5xl font-bold text-center mb-4 text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500 bg-[length:200%_auto] animate-gradient-pan">
            Model Dashboard
          </h1>
          <p className="text-center text-gray-400 mb-10 text-base">
            A technical deep-dive into the DiabRisk AI model, its performance, and the data it was trained on.
          </p>
        </motion.div>

        {/* --- Key Metrics (Final Model) --- */}
        <h3 className="text-2xl font-semibold mb-4 text-gray-100">Final Model Metrics (V3 - Tuned)</h3>
        <motion.div
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
          initial="hidden"
          animate="visible"
          variants={{
            visible: { transition: { staggerChildren: 0.1 } }
          }}
        >
          <MetricCard title="Model" value="XGBoost (Tuned)" icon={<Zap size={20} className="text-white"/>} />
          <MetricCard title="ROC AUC" value="0.867" icon={<CheckCircle size={20} className="text-white"/>} />
          <MetricCard title="Accuracy" value="73.4%" icon={<Info size={20} className="text-white"/>} />
          <MetricCard title="Diabetic Recall" value="88.0%" icon={<CheckCircle size={20} className="text-white"/>} />
        </motion.div>

        {/* --- Model Features --- */}
        <motion.div
          className="bg-black/70 backdrop-blur-md rounded-lg shadow-2xl border border-gray-800/50 p-6 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <h3 className="text-xl font-semibold mb-4 flex items-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
            <LayoutGrid className="mr-2" />
            Model Features (17 Inputs)
          </h3>
          <p className="text-sm text-gray-400 mb-4">The model was trained on these 17 "leak-free" features. The API can impute missing values.</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-2 text-gray-300 text-sm">
            <div>
              <h4 className="font-semibold text-purple-300 mt-2">Demographics</h4>
              <ul>
                <li>AGE</li>
                <li>AGE_BIN</li>
                <li>SEX</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-purple-300 mt-2">Body Composition</h4>
              <ul>
                <li>BMI</li>
                <li>BMI_CLASS</li>
                <li>WAIST_HT_RATIO</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-purple-300 mt-2">Blood Pressure</h4>
              <ul>
                <li>SBP_MEAN</li>
                <li>DBP_MEAN</li>
                <li>HYPERTENSION_FLAG</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-purple-300 mt-2">Lipid Panel</h4>
              <ul>
                <li>TCHOL</li>
                <li>HDL</li>
                <li>TRIG</li>
                <li>CHOL_HDL_RATIO</li>
                <li>TG_TO_HDL</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-purple-300 mt-2">Other Labs</h4>
              <ul>
                <li>ACR (Urine)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-purple-300 mt-2">Lifestyle</h4>
              <ul>
                <li>SMOKER</li>
                <li>PA_SCORE</li>
              </ul>
            </div>
          </div>
        </motion.div>

        {/* --- Charts --- */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <ChartCard title="Feature Importance (Simulated SHAP)" icon={<BarChart2 />}>
            <ResponsiveContainer width="100%" height="100%">
              {/* --- 2. BAR CHART FIX --- */}
              <BarChart data={featureImportanceData} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
                <XAxis type="number" stroke="#9ca3af" tick={{ fill: '#d1d5db' }} />
                <YAxis
                  dataKey="name"
                  type="category"
                  stroke="#9ca3af"
                  tick={{ fill: '#d1d5db' }}
                  width={100} // Give space for labels
                  textAnchor="end" // Align labels
                  interval={0} // Show all labels
                />
                <Tooltip
                  cursor={{ fill: 'rgba(255, 255, 255, 0.1)' }}
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.5rem' }}
                  labelStyle={{ color: '#f3f4f6' }}
                />
                <Bar dataKey="importance" fill="#c084fc" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Class Distribution (Test Set)" icon={<PieIcon />}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={classDistributionData}
                  cx="50%"
                  cy="50%"
                  outerRadius="80%" // <-- 3. PIE CHART FIX
                  dataKey="value"
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                >
                  {classDistributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Legend
                  wrapperStyle={{ color: '#d1d5db' }}
                  formatter={(value, entry) => <span style={{ color: '#d1d5db' }}>{value}</span>}
                />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* --- Scatterplot (now with 250 points) --- */}
        <ChartCard title="Data Visualization (Simulated)" icon={<Users />}>
          <h4 className="text-sm -mt-4 mb-4 text-gray-400">Relationship between Age and BMI, colored by class. This helps visualize the model's decision boundary.</h4>
          <ResponsiveContainer width="100%" height="100%" className="text-white">
            <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
              <XAxis
                dataKey="age"
                type="number"
                name="Age"
                unit=" yrs"
                stroke="#9ca3af"
                tick={{ fill: '#d1d5db' }}
              />
              <YAxis
                dataKey="bmi"
                type="number"
                name="BMI"
                stroke="#9ca3af"
                tick={{ fill: '#d1d5db' }}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3', stroke: '#a855f7' }}
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.5rem' }}
                labelStyle={{ color: '#f3f4f6' }}
              />
              <Legend />
              <Scatter name="Non-Diabetic" data={nonDiabeticData} fill="#3b82f6" opacity={0.7} />
              <Scatter name="Diabetic" data={diabeticData} fill="#f472b6" opacity={0.7} />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>


        {/* --- Classification Report & Confusion Matrix (Final Model) --- */}
        <motion.div
          className="bg-black/70 backdrop-blur-md rounded-lg shadow-2xl border border-gray-800/50 p-6 mt-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <h3 className="text-xl font-semibold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
            Final Model Performance (V3)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-200 mb-2">Classification Report</h4>
              {/* --- 1. SCROLLBAR FIX --- */}
              <pre className="p-3 bg-gray-900/50 rounded-lg text-sm text-gray-300 overflow-x-auto dark-scrollbar">
                {`                precision    recall  f1-score   support

  0 (Non-Diabetic)       0.97      0.71      0.82      1960
     1 (Diabetic)       0.34      0.88      0.49       337

       accuracy                           0.73      2297
      macro avg       0.66      0.80      0.66      2297
   weighted avg       0.88      0.73      0.77      2297`}
              </pre>
            </div>
            <div>
              <h4 className="font-semibold text-gray-200 mb-2">Confusion Matrix</h4>
              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="p-3 bg-green-900/40 rounded-lg border border-green-700">
                  <p className="text-xs text-green-300">True Negative</p>
                  <p className="text-2xl font-bold">1390</p>
                </div>
                <div className="p-3 bg-red-900/40 rounded-lg border border-red-700">
                  <p className="text-xs text-red-300">False Positive (FP)</p>
                  <p className="text-2xl font-bold">570</p>
                </div>
                <div className="p-3 bg-red-900/40 rounded-lg border border-red-700">
                  <p className="text-xs text-red-300">False Negative (FN)</p>
                  <p className="text-2xl font-bold">40</p>
                </div>
                <div className="p-3 bg-green-900/40 rounded-lg border border-green-700">
                  <p className="text-xs text-green-300">True Positive</p>
                  <p className="text-2xl font-bold">297</p>
                </div>
              </div>
              <p className="text-xs text-gray-400 mt-3">
                <strong>Interpretation:</strong> The model is tuned for high **Recall (88%)**. This means it correctly identifies 297/337 diabetic cases, missing only **40**. This is a massive improvement from the 150 missed cases previously.
              </p>
              <p className="text-xs text-gray-400 mt-2">
                <strong>The Trade-Off:</strong> To achieve this, precision is lower (34%), meaning we have more False Positives (570). For a medical screening tool, this is the correct, safer trade-off.
              </p>
            </div>
          </div>
        </motion.div>

        {/* --- NEW: Model Iteration & Comparison --- */}
        <motion.div
          className="bg-black/70 backdrop-blur-md rounded-lg shadow-2xl border border-gray-800/50 p-6 mt-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <h3 className="text-xl font-semibold mb-4 flex items-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
            <BookOpen className="mr-2" />
            Model Iteration & Research Comparison
          </h3>
          <p className="text-sm text-gray-400 mb-4">
            A model's final metrics are only part of the story. This table shows the project's evolution, from correcting data leakage to tuning for a specific clinical goal (High Recall).
          </p>
          <div className="overflow-x-auto dark-scrollbar">
            <table className="w-full text-left text-gray-300 text-sm">
              <thead className="border-b border-gray-700">
                <tr>
                  <th className="pb-2 font-semibold">Model Version</th>
                  <th className="pb-2 font-semibold">AUC Score</th>
                  <th className="pb-2 font-semibold">Diabetic Recall</th>
                  <th className="pb-2 font-semibold">Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-800">
                  <td className="py-2 text-purple-300 font-bold">V3: Tuned Model (Live)</td>
                  <td className="py-2 font-bold">0.867</td>
                  <td className="py-2 font-bold">88.0%</td>
                  <td className="py-2">
                    <span className="flex items-center"><CheckCircle size={14} className="text-green-500 mr-1.5" />Optimized for safety (high recall).</span>
                  </td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-2">V2: Baseline Model</td>
                  <td className="py-2">0.852</td>
                  <td className="py-2">58.0%</td>
                  <td className="py-2">
                    <span className="flex items-center"><Info size={14} className="text-blue-500 mr-1.5" />Good baseline, but missed 150 cases (low recall).</span>
                  </td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-2">V1: Leaky Model</td>
                  <td className="py-2">0.999</td>
                  <td className="py-2">99.5%</td>
                  <td className="py-2">
                    <span className="flex items-center"><AlertTriangle size={14} className="text-red-500 mr-1.5" />Textbook data leakage. Unusable.</span>
                  </td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-2">
                    <a
                      href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11931972/"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:underline"
                    >
                      Published ML Research (Avg.)
                    </a>
                  </td>
                  <td className="py-2">0.85 - 0.88</td>
                  <td className="py-2">N/A</td>
                  <td className="py-2">Our model is highly competitive.</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-2">
                    <a
                      href="https://www.mdcalc.com/calc/4000/findrisc-finnish-diabetes-risk-score"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:underline"
                    >
                      Traditional Score (e.g., FINDRISC)
                    </a>
                  </td>
                  <td className="py-2">~0.80 - 0.85</td>
                  <td className="py-2">N/A</td>
                  <td className="py-2">Our model outperforms standard clinical tools.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </main>
  );
}
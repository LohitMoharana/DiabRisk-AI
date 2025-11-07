"use client";

import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import {
  Droplet,
  HeartPulse,
  Activity,
  Scale,
  Cigarette,
  AlertTriangle,
  FileText,
  User,
  CheckCircle2,
  ChevronDown,
  Info,
  Zap,
  Loader2
} from 'lucide-react';


const Tooltip = ({ text, children }) => {
  const [isHovered, setIsHovered] = useState(false);
  return (
    <div className="relative flex items-center">
      {children}
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 5 }}
            transition={{ duration: 0.2 }}
            className="absolute left-1/2 -translate-x-1/2 top-full mt-2 w-48 p-2 bg-gray-800 text-white text-xs rounded-lg shadow-lg z-20 border border-gray-700"
          >
            {text}
          </motion.div>
        )}
      </AnimatePresence>
      <div
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className="ml-1.5 cursor-help"
      >
        <Info size={14} className="text-gray-400 hover:text-purple-400 transition-colors" />
      </div>
    </div>
  );
};

const Input = ({ label, name, type = "number", value, onChange, placeholder, icon, tooltip }) => (
  <div className="relative w-full">
    <label className="block text-xs uppercase tracking-wider font-medium text-gray-400 mb-1.5 flex items-center">
      {tooltip ? <Tooltip text={tooltip}>{label}</Tooltip> : label}
    </label>
    <div className="absolute left-3 top-9 text-gray-400">{icon}</div>
    <input
      name={name}
      type={type}
      value={value === null ? '' : value}
      onChange={onChange}
      placeholder={placeholder}
      className="w-full pl-10 pr-4 py-2 bg-black/50 text-gray-100 border border-gray-700 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-300 focus:shadow-[0_0_15px_rgba(192,132,252,0.3)] font-sans"
    />
  </div>
);

const Select = ({ label, name, value, onChange, children, icon, tooltip }) => (
  <div className="relative w-full">
    <label className="block text-xs uppercase tracking-wider font-medium text-gray-400 mb-1.5 flex items-center">
      {tooltip ? <Tooltip text={tooltip}>{label}</Tooltip> : label}
    </label>
    <div className="absolute left-3 top-9 text-gray-400">{icon}</div>
    <select
      name={name}
      value={value === null ? '' : value}
      onChange={onChange}
      className="w-full pl-10 pr-10 py-2 appearance-none bg-black/50 text-gray-100 border border-gray-700 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-300 focus:shadow-[0_0_15px_rgba(192,132,252,0.3)] font-sans"
    >
      {children}
    </select>
    <ChevronDown className="absolute right-3 top-9 text-gray-400 w-5 h-5 pointer-events-none" />
  </div>
);

// --- 2. Risk Gauge Component ---
const RiskGauge = ({ probability }) => {
  const percentage = Math.round(probability * 100);
  // Updated colors to match the theme
  let color = '#4ade80'; // Green (Good)
  if (percentage >= 20) color = '#c084fc'; // Purple (Moderate)
  if (percentage >= 50) color = '#f472b6'; // Pink (High)

  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <div className="w-40 h-40 mx-auto flex items-center justify-center relative">
      <svg
        className="transform -rotate-90"
        width="100%"
        height="100%"
        viewBox="0 0 120 120"
      >
        <circle
          className="text-gray-800"
          strokeWidth="12"
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx="60"
          cy="60"
        />
        <motion.circle
          strokeWidth="12"
          stroke={color}
          fill="transparent"
          r={radius}
          cx="60"
          cy="60"
          strokeDasharray={circumference}
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.5, ease: "easeOut" }}
        />
      </svg>
      <motion.span
        className="absolute text-3xl font-bold"
        style={{ color: color }}
        initial={{ opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.5, duration: 0.5 }}
      >
        {percentage}%
      </motion.span>
    </div>
  );
};

// --- 3. Results Card Component ---
const ResultsCard = ({ result, onReset }) => {
  const { risk_probability, risk_level, advice } = result;

  const topicIcons = {
    "Urgent": <AlertTriangle className="text-pink-400" />,
    "Weight Management": <Scale className="text-purple-400" />,
    "Blood Pressure": <HeartPulse className="text-pink-500" />,
    "Physical Activity": <Activity className="text-green-400" />,
    "Smoking": <Cigarette className="text-gray-400" />,
    "Cholesterol & Fats": <Droplet className="text-purple-300" />,
    "General": <CheckCircle2 className="text-blue-400" />,
  };

  const cardVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: { opacity: 1, scale: 1, transition: { duration: 0.5 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 }
  };

  return (
    <motion.div
      className="w-full max-w-2xl p-6 bg-black/70 backdrop-blur-md rounded-lg shadow-xl border border-gray-800/50"
      variants={cardVariants}
      initial="hidden"
      animate="visible"
    >
      <h2 className="text-2xl font-bold text-center text-gray-100 mb-4">Your Risk Assessment</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        <motion.div
          className="flex flex-col items-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          <RiskGauge probability={risk_probability} />
          <p className={`text-3xl font-bold mt-4 ${
            risk_level === 'High' ? 'text-pink-400' :
            risk_level === 'Moderate' ? 'text-purple-400' : 'text-green-400'
          }`}>
            {risk_level} Risk
          </p>
        </motion.div>
        <motion.div
          className="flex flex-col"
        >
          <h3 className="text-lg font-semibold mb-2 text-gray-200">Personalized Advice</h3>
          <motion.ul
            className="space-y-3"
            initial="hidden"
            animate="visible"
            transition={{ staggerChildren: 0.1, delayChildren: 0.3 }}
          >
            {advice.map((item, index) => (
              <motion.li
                key={index}
                className="flex items-start space-x-3 p-2 rounded-lg transition-all duration-300 hover:bg-gray-800/80 hover:shadow-lg"
                variants={itemVariants}
                whileHover={{ scale: 1.02 }}
              >
                <span className="flex-shrink-0 w-6 h-6 mt-1">
                  {topicIcons[item.topic] || <FileText className="text-gray-400" />}
                </span>
                <div>
                  <p className="font-semibold text-gray-100">{item.topic}</p>
                  <p className="text-sm text-gray-300">{item.message}</p>
                </div>
              </motion.li>
            ))}
          </motion.ul>
        </motion.div>
      </div>
      <motion.button
        onClick={onReset}
        className="w-full mt-8 px-4 py-2 bg-gradient-to-r from-pink-500 via-purple-500 to-blue-600 text-white font-semibold rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-gray-950 bg-[length:200%_auto] animate-gradient-pan"
        whileHover={{ scale: 1.02,
           boxShadow: "0px 0px 15px rgba(192, 132, 252, 0.5)" }}
        whileTap={{ scale: 0.98 }}
      >
        Run a New Assessment
      </motion.button>
    </motion.div>
  );
};

// --- 4. Dynamic Background Glow Component ---
const AnimatedGlow = ({ className }) => <div className={className} />;

const getBackgroundGlows = (risk) => {
  if (risk === 'High') {
    return (
      <>
        <AnimatedGlow className="absolute top-0 left-0 w-96 h-96 bg-pink-600/30 rounded-full blur-[150px] opacity-50" />
        <AnimatedGlow className="absolute bottom-0 right-0 w-96 h-96 bg-red-500/30 rounded-full blur-[150px] opacity-50" />
        <AnimatedGlow className="absolute top-1/3 left-1/3 w-80 h-80 bg-purple-500/20 rounded-full blur-[120px] opacity-40" />
      </>
    );
  }
  if (risk === 'Moderate') {
    return (
      <>
        <AnimatedGlow className="absolute top-0 left-0 w-96 h-96 bg-purple-600/30 rounded-full blur-[150px] opacity-50" />
        <AnimatedGlow className="absolute bottom-0 right-0 w-96 h-96 bg-blue-500/30 rounded-full blur-[150px] opacity-50" />
        <AnimatedGlow className="absolute top-1/3 left-1/3 w-80 h-80 bg-indigo-500/20 rounded-full blur-[120px] opacity-40" />
      </>
    );
  }
  if (risk === 'Low') {
    return (
      <>
        <AnimatedGlow className="absolute top-0 left-0 w-96 h-96 bg-green-600/30 rounded-full blur-[150px] opacity-50" />
        <AnimatedGlow className="absolute bottom-0 right-0 w-96 h-96 bg-teal-500/30 rounded-full blur-[150px] opacity-50" />
        <AnimatedGlow className="absolute top-1/3 left-1/3 w-80 h-80 bg-blue-500/20 rounded-full blur-[120px] opacity-40" />
      </>
    );
  }
  // Default (no result)
  return (
    <>
      <AnimatedGlow className="absolute top-0 left-0 w-96 h-96 bg-purple-600/30 rounded-full blur-[150px] opacity-50" />
      <AnimatedGlow className="absolute bottom-0 right-0 w-96 h-96 bg-pink-500/30 rounded-full blur-[150px] opacity-50" />
      <AnimatedGlow className="absolute top-1/3 left-1/3 w-80 h-80 bg-blue-500/20 rounded-full blur-[120px] opacity-40" />
    </>
  );
};


// --- 5. Main App Component (page.jsx) ---
export default function Home() {
  const [formData, setFormData] = useState({
    AGE: null, AGE_BIN: null, SEX: null, BMI: null, BMI_CLASS: null,
    WAIST_HT_RATIO: null, SBP_MEAN: null, DBP_MEAN: null, HYPERTENSION_FLAG: null,
    TCHOL: null, HDL: null, TRIG: null, CHOL_HDL_RATIO: null, TG_TO_HDL: null,
    ACR: null, SMOKER: null, PA_SCORE: null,
  });

  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- ROBUST FORM HANDLERS ---
  const handleChange = (e) => {
    const { name, value, type } = e.target;
    let processedValue;
    if (value === 'null' || value === '') {
      processedValue = null;
    } else if (type === 'number') {
      processedValue = parseFloat(value);
      if (isNaN(processedValue)) processedValue = null;
    } else if (['SEX', 'HYPERTENSION_FLAG', 'SMOKER'].includes(name)) {
      processedValue = parseInt(value, 10);
      if (isNaN(processedValue)) processedValue = null;
    } else {
      processedValue = value;
    }
    setFormData(prev => ({ ...prev, [name]: processedValue }));
  };

  const handleAgeChange = (e) => {
    const { value } = e.target;
    const age = value === '' ? null : parseFloat(value);
    let ageBin = null;
    if (age !== null && !isNaN(age)) {
      if (age < 30) ageBin = "<30";
      else if (age <= 39) ageBin = "30-39";
      else if (age <= 49) ageBin = "40-49";
      else if (age <= 59) ageBin = "50-59";
      else if (age > 59) ageBin = "60+";
    }
    setFormData(prev => ({ ...prev, AGE: age, AGE_BIN: ageBin }));
  };

  const handleBmiChange = (e) => {
    const { value } = e.target;
    const bmi = value === '' ? null : parseFloat(value);
    let bmiClass = null;
    if (bmi !== null && !isNaN(bmi)) {
      if (bmi < 18.5) bmiClass = "underweight";
      else if (bmi < 25) bmiClass = "normal";
      else if (bmi < 30) bmiClass = "overweight";
      else if (bmi >= 30) bmiClass = "Obese";
    }
    setFormData(prev => ({ ...prev, BMI: bmi, BMI_CLASS: bmiClass }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/predict`  // For Vercel (Production)
  : "http://127.0.0.1:5000/predict";           // For Localhost (Development)
    try {
      // Fake delay for demo purposes
      // await new Promise(res => setTimeout(res, 1500));
      const response = await axios.post(apiUrl, formData);
      setResult(response.data);
    } catch (err) {
      console.error(err);
      const errMsg = err.response?.data?.detail || "A network error occurred. Please try again.";
      setError(errMsg);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setFormData({
      AGE: null, AGE_BIN: null, SEX: null, BMI: null, BMI_CLASS: null,
      WAIST_HT_RATIO: null, SBP_MEAN: null, DBP_MEAN: null, HYPERTENSION_FLAG: null,
      TCHOL: null, HDL: null, TRIG: null, CHOL_HDL_RATIO: null, TG_TO_HDL: null,
      ACR: null, SMOKER: null, PA_SCORE: null,
    });
  };

  // --- Animation Variants (FASTER) ---
  const formContainerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.07 } } // Faster stagger
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: "easeOut" } } // Faster item animation
  };

  // Get the current risk level for the background
  const riskLevel = result?.risk_level || null;

  return (
    <main className="flex flex-col items-center min-h-screen p-4 md:p-8 bg-black text-gray-200 overflow-y-auto relative font-sora">
      {/* --- CSS FIXES, NEW ANIMATIONS, & FONT --- */}
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');

        .font-sora {
          font-family: 'Sora', sans-serif;
        }

        /* Hide scrollbar arrows for number inputs */
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }
        input[type=number] {
          -moz-appearance: textfield;
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
      `}</style>

      {/* --- Animated Background Glow (FIXED & DYNAMIC) --- */}
      <div className="fixed inset-0 z-0 overflow-hidden">
        <AnimatePresence>
          <motion.div
            key={riskLevel} // This will trigger the animation on change
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 1.0 }}
            className="absolute inset-0"
          >
            {getBackgroundGlows(riskLevel)}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* --- Main Content (Scrollable) --- */}
      <div className="w-full max-w-4xl z-10 py-10">
        <motion.h1
          className="text-4xl md:text-5xl font-bold text-center mb-4 text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-purple-500 bg-[length:200%_auto] animate-gradient-pan"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          DiabRisk AI
        </motion.h1>
        <motion.p
          className="text-center text-gray-300 mb-10 text-base"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }} // Faster
        >
          Early Diabetes Risk Predictor
        </motion.p>

        {/* Conditional Rendering: Show Results or Form */}
        <AnimatePresence mode="wait">
          {!result ? (
            <motion.form
              key="form"
              onSubmit={handleSubmit}
              className="p-6 md:p-8 bg-black/70 backdrop-blur-md rounded-lg shadow-2xl border border-gray-800/50"
              variants={formContainerVariants}
              initial="hidden"
              animate="visible"
              exit={{ opacity: 0, scale: 0.95 }}
            >
              <h2 className="text-2xl font-semibold text-gray-100 mb-6 border-b border-gray-700 pb-3 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                Enter Your Health Metrics
              </h2>

              <p className="text-sm text-gray-400 mb-6">
                Please fill in as many fields as you can. Our AI can impute (predict) missing values, but more data leads to a more accurate result.
              </p>

              {/* Form Sections */}
              <div className="space-y-8">
                <motion.div
                  variants={itemVariants}
                  className="p-4 bg-gray-900/30 border border-gray-800/50 rounded-lg transition-all duration-300 hover:border-purple-500/50"
                  whileHover={{ scale: 1.015, borderColor: '#a855f7', boxShadow: '0 0 20px rgba(168, 85, 247, 0.2)' }}
                >
                  <h3 className="text-lg font-semibold mb-4 flex items-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400"><User className="mr-2 text-purple-400" />Demographics</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Input label="Age" name="AGE" value={formData.AGE} onChange={handleAgeChange} placeholder="e.g., 55" icon={<User size={18} />} />
                    <Select label="Sex" name="SEX" value={formData.SEX} onChange={handleChange} icon={<User size={18} />}>
                      <option value="null">Select...</option>
                      <option value="1">Male</option>
                      <option value="2">Female</option>
                    </Select>
                  </div>
                </motion.div>

                <motion.div
                  variants={itemVariants}
                  className="p-4 bg-gray-900/30 border border-gray-800/50 rounded-lg transition-all duration-300 hover:border-purple-500/50"
                  whileHover={{ scale: 1.015, borderColor: '#a855f7', boxShadow: '0 0 20px rgba(168, 85, 247, 0.2)' }}
                >
                  <h3 className="text-lg font-semibold mb-4 flex items-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400"><Scale className="mr-2 text-purple-400" />Body Composition</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Input
                      label="Body Mass Index (BMI)"
                      name="BMI"
                      value={formData.BMI}
                      onChange={handleBmiChange}
                      placeholder="e.g., 29.5"
                      icon={<Scale size={18} />}
                      tooltip="Calculated as weight (kg) / height (m)². Don't worry if you don't know, our AI can estimate."
                    />
                    <Input
                      label="Waist-to-Height Ratio"
                      name="WAIST_HT_RATIO"
                      value={formData.WAIST_HT_RATIO}
                      onChange={handleChange}
                      placeholder="e.g., 0.58"
                      icon={<Scale size={18} />}
                      tooltip="Your waist circumference divided by your height. A value > 0.5 is a high-risk indicator."
                    />
                  </div>
                </motion.div>

                <motion.div
                  variants={itemVariants}
                  className="p-4 bg-gray-900/30 border border-gray-800/50 rounded-lg transition-all duration-300 hover:border-purple-500/50"
                  whileHover={{ scale: 1.015, borderColor: '#a855f7', boxShadow: '0 0 20px rgba(168, 85, 247, 0.2)' }}
                >
                  <h3 className="text-lg font-semibold mb-4 flex items-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400"><HeartPulse className="mr-2 text-purple-400" />Blood Pressure</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Input label="Systolic (Top #)" name="SBP_MEAN" value={formData.SBP_MEAN} onChange={handleChange} placeholder="e.g., 135" icon={<HeartPulse size={18} />} />
                    <Input label="Diastolic (Bottom #)" name="DBP_MEAN" value={formData.DBP_MEAN} onChange={handleChange} placeholder="e.g., 88" icon={<HeartPulse size={18} />} />
                    <Select label="Hypertension" name="HYPERTENSION_FLAG" value={formData.HYPERTENSION_FLAG} onChange={handleChange} icon={<HeartPulse size={18} />}>
                      <option value="null">Select...</option>
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </Select>
                  </div>
                </motion.div>

                <motion.div
                  variants={itemVariants}
                  className="p-4 bg-gray-900/30 border border-gray-800/50 rounded-lg transition-all duration-300 hover:border-purple-500/50"
                  whileHover={{ scale: 1.015, borderColor: '#a855f7', boxShadow: '0 0 20px rgba(168, 85, 247, 0.2)' }}
                >
                  <h3 className="text-lg font-semibold mb-4 flex items-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400"><Droplet className="mr-2 text-purple-400" />Lab Results (Optional)</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Input label="Total Cholesterol" name="TCHOL" value={formData.TCHOL} onChange={handleChange} placeholder="e.g., 210" icon={<Droplet size={18} />} />
                    <Input label="HDL ('Good' Chol.)" name="HDL" value={formData.HDL} onChange={handleChange} placeholder="e.g., 45" icon={<Droplet size={18} />} />
                    <Input label="Triglycerides" name="TRIG" value={formData.TRIG} onChange={handleChange} placeholder="e.g., 150" icon={<Droplet size={18} />} />
                    <Input
                      label="Chol/HDL Ratio"
                      name="CHOL_HDL_RATIO"
                      value={formData.CHOL_HDL_RATIO}
                      onChange={handleChange}
                      placeholder="e.g., 4.6"
                      icon={<Droplet size={18} />}
                      tooltip="Total Cholesterol divided by HDL. Lower is better."
                    />
                    <Input
                      label="Trig/HDL Ratio"
                      name="TG_TO_HDL"
                      value={formData.TG_TO_HDL}
                      onChange={handleChange}
                      placeholder="e.g., 3.3"
                      icon={<Droplet size={18} />}
                      tooltip="Triglycerides divided by HDL. A strong indicator of insulin resistance. Lower is better."
                    />
                    <Input
                      label="ACR (Urine)"
                      name="ACR"
                      value={formData.ACR}
                      onChange={handleChange}
                      placeholder="e.g., 15"
                      icon={<Droplet size={18} />}
                      tooltip="Albumin-to-Creatinine Ratio. Measures early kidney damage."
                    />
                  </div>
                </motion.div>

                <motion.div
                  variants={itemVariants}
                  className="p-4 bg-gray-900/30 border border-gray-800/50 rounded-lg transition-all duration-300 hover:border-purple-500/50"
                  whileHover={{ scale: 1.015, borderColor: '#a855f7', boxShadow: '0 0 20px rgba(168, 85, 247, 0.2)' }}
                >
                  <h3 className="text-lg font-semibold mb-4 flex items-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400"><Activity className="mr-2 text-purple-400" />Lifestyle</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Select label="Smoking Status" name="SMOKER" value={formData.SMOKER} onChange={handleChange} icon={<Cigarette size={18} />}>
                      <option value="null">Select...</option>
                      <option value="0">Never Smoked</option>
                      <option value="1">Current Smoker</option>
                      <option value="2">Former Smoker</option>
                    </Select>
                    <Input
                      label="Physical Activity (PA) Score"
                      name="PA_SCORE"
                      value={formData.PA_SCORE}
                      onChange={handleChange}
                      placeholder="e.g., 100"
                      icon={<Activity size={18} />}
                      tooltip="A score based on your weekly exercise. More is better. (MET-minutes/week)"
                    />
                  </div>
                </motion.div>
              </div>

              {/* Error Message */}
              {error && (
                <motion.div
                  className="mt-6 text-center text-pink-400 font-semibold bg-pink-900/20 p-3 rounded-lg border border-pink-700"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  {error}
                </motion.div>
              )}

              {/* Submit Button */}
              <motion.button
                type="submit"
                disabled={isLoading}
                className="w-full mt-8 px-6 py-3 bg-gradient-to-r from-pink-500 via-purple-500 to-blue-600 text-white font-bold rounded-lg shadow-lg focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 focus:ring-offset-gray-950 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 bg-[length:200%_auto] animate-gradient-pan"
                whileHover={{ scale: 1.02, boxShadow: "0px 0px 20px rgba(192, 132, 252, 0.5)" }}
                whileTap={{ scale: 0.98 }}
              >
                {isLoading ? (
                  <Loader2 className="animate-spin" />
                ) : (
                  <Zap size={18} />
                )}
                <span>{isLoading ? 'Analyzing...' : 'Calculate My Risk'}</span>
              </motion.button>
            </motion.form>
          ) : (
            <motion.div key="results" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
              <ResultsCard result={result} onReset={handleReset} />
            </motion.div>
          )}
        </AnimatePresence>

        <footer className="mt-8 text-center text-gray-500 text-sm">
  <p>
    © {new Date().getFullYear()} DiabRisk AI · Powered by FastAPI + Next.js
  </p>
  <p className="mt-2">
    <Link href="/dashboard" className="text-purple-400 hover:text-purple-300 transition-colors hover:underline">
      View Model Details & Dashboard
    </Link>
  </p>
</footer>
    </div>
    </main>
  );
}
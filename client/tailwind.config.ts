import type { Config } from "tailwindcss";

// In Tailwind CSS v4, the design tokens live in globals.css inside @theme {}.
// This file is only needed to configure content scanning paths.
const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
};

export default config;

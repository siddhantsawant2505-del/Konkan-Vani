import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Konkan Vani — Preserving Coastal Wisdom",
    template: "%s | Konkan Vani",
  },
  description:
    "A living repository of Konkani idioms, proverbs, and sayings — translated, contextualized, and preserved for future generations.",
  keywords: ["Konkani", "idioms", "proverbs", "Konkan", "language preservation", "NLP"],
  authors: [{ name: "Konkan Vani Project" }],
  openGraph: {
    title: "Konkan Vani — Preserving Coastal Wisdom",
    description:
      "Search thousands of Konkani idioms in Devanagari, Kannada, or Roman script.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="light">
      <head>
        {/* Google Fonts: preconnect */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />

        {/* EB Garamond (serif — idiom display text) */}
        <link
          href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,500;1,600&display=swap"
          rel="stylesheet"
        />

        {/* Hanken Grotesk (sans — UI labels and body text) */}
        <link
          href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap"
          rel="stylesheet"
        />

        {/* Material Symbols Outlined (icons) */}
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="azulejos-pattern">
        {children}
      </body>
    </html>
  );
}

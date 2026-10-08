import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Satellite Super-Resolution Lab",
  description:
    "Interactive research platform for Real-ESRGAN and diffusion-based satellite image super-resolution."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

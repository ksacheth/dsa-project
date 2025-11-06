"use client";
import Image from "next/image";
import { useRef, useState } from "react";
import axios from "axios";


export default function Home() {
  const url = useRef();
  const [shortenedURL, setShortenedURL] = useState("");
  const [error, setError] = useState("");
  const [strategy, setStrategy] = useState("separate_chaining");

  const strategies = [
    "separate_chaining",
    "linear_probing",
    "double_hashing",
    "cuckoo",
  ];

  async function sendURL(){
    if (!url.current.value) {
      setError("Please enter a URL.");
      return;
    }
    setError("");
    setShortenedURL("");

    try {
      const formData = new FormData();
      formData.append('url', url.current.value);
      formData.append('expiry', '3600');
      formData.append('strategy', strategy);

      const response = await axios.post("http://localhost:8000/shorten", formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.data && response.data.short_url) {
        setShortenedURL(response.data.short_url);
      }
    } catch (err) {
      console.error("Error shortening URL:", err);
      setError("Failed to shorten URL. Please check the console for details.");
    }
  }

  return (
    <div>
      <div className="text-6xl font-bold text-center mt-[4rem]"  >
          URL Shortner
      </div>
      <div className="flex flex-col items-center justify-center mt-[5rem]">
          <input
            ref={url}
            placeholder="Enter the url..."
            className="border rounded-2xl border-gray-300 w-[25rem] px-[2rem] py-[1rem] outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-colors"
          />
          <div className="mt-4 text-center">
            <span className="mr-4 font-medium">Collision Strategy:</span>
            <div className="inline-flex rounded-md shadow-sm mt-2" role="group">
              {strategies.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setStrategy(s)}
                  className={`py-2 px-4 text-sm font-medium transition-colors ${
                    strategy === s
                      ? 'bg-blue-500 text-white'
                      : 'bg-white text-gray-900 hover:bg-gray-100'
                  } border border-gray-200 first:rounded-l-lg last:rounded-r-lg focus:z-10 focus:ring-2 focus:ring-blue-500`}
                >
                  {s.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </button>
              ))}
            </div>
          </div>
          <button onClick={sendURL} className="mt-[2rem] bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition-colors">
            Shorten
          </button>
          {shortenedURL && (
            <div className="mt-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
              Shortened URL: <a href={shortenedURL} target="_blank" rel="noopener noreferrer" className="font-bold">{shortenedURL}</a>
            </div>
          )}
          {error && (
            <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
              {error}
            </div>
          )}
      </div>
    </div>
  );
}

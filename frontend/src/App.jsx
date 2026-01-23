import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Results from "./pages/Results";
import Docs from "./pages/Docs";
import API from "./pages/API";


export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/results" element={<Results />} />
      <Route path="/docs" element={<Docs />} />
      <Route path="/api" element={<API />} />
    </Routes>
  );
}

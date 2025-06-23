fetch("https://f412-34-53-34-52.ngrok-free.app/generate")
  .then(response => response.json())
  .then(data => {
    console.log("API Response:", data);
    // setGeneratedTestCases(data.generatedTestCases);
    // setSimilarExamples(data.similarExamples);
  })
  .catch(error => console.error("Fetch error:", error));
async function startStream() {
    showSpinner("Starting detection...");
    const response = await fetch('/start_stream', {
        method: 'POST'
    });

    if (response.ok) {
        await loadGallery();
        hideSpinner();
        alert("Detection complete! Select a person to match.");
    } else {
        hideSpinner();
        alert("Failed to start detection.");
    }
}


async function loadGallery() {
  const res = await fetch('/get_query_images');
  const images = await res.json();
  const gallery = document.getElementById("gallery");
  gallery.innerHTML = "";

  const perPage = 9;
  const totalPages = Math.ceil(images.length / perPage);

  for (let page = 0; page < totalPages; page++) {
    const pageContainer = document.createElement("div");
    pageContainer.className = "gallery-page";
    pageContainer.style.display = "grid";
    pageContainer.style.gridTemplateColumns = "repeat(3, 1fr)";
    pageContainer.style.gridGap = "12px";
    pageContainer.style.minWidth = "360px"; // width of 3 images + gaps
    pageContainer.style.flex = "0 0 auto";  // keep fixed page width

    const pageImages = images.slice(page * perPage, (page + 1) * perPage);

    pageImages.forEach(filename => {
      const img = document.createElement("img");
      img.src = `/query_images/${filename}`;
      img.style.width = "100px";
      img.style.height = "140px";
      img.style.objectFit = "cover";
      img.style.border = "2px solid transparent";
      img.style.borderRadius = "6px";
      img.style.cursor = "pointer";
      img.style.transition = "transform 0.2s ease";

      img.onclick = () => {
  selectImage(filename, img);
};

      img.onmouseenter = () => img.style.transform = "scale(1.05)";
      img.onmouseleave = () => img.style.transform = "scale(1)";

      pageContainer.appendChild(img);
    });

    gallery.appendChild(pageContainer);
  }
}


async function selectImage(filename, imgElement) {
  await fetch(`/select_query/${filename}`, { method: 'POST' });

  document.querySelectorAll("#gallery img").forEach(img => {
    img.style.border = "2px solid transparent";
  });
  imgElement.style.border = "3px solid #3498db";

  const preview = document.getElementById("selected-preview");
  preview.innerHTML = `
    <h3>Selected Person</h3>
    <img src="/query/query.jpg?ts=${Date.now()}" width="150" style="border-radius: 8px;" />
  `;

  // Start matching and WebSocket stream
  await fetch('/start_matching');
  startWebSocketStream();
}





async function resetSession() {
    showSpinner("Resetting session...");
    const res = await fetch('/reset', { method: 'POST' });
    if (res.ok) {
        document.getElementById("gallery").innerHTML = "";
        document.getElementById("selected-preview").innerHTML = "";
        hideSpinner();
        alert("Session reset.");
    } else {
        hideSpinner();
        alert("Failed to reset.");
    }
}

function showSpinner(msg = "Loading...") {
    const spinner = document.getElementById("spinner");
    spinner.style.display = "block";
    spinner.querySelector("br").nextSibling.textContent = msg;
}

function hideSpinner() {
    document.getElementById("spinner").style.display = "none";
}


// function scrollGallery(direction) {
//     const gallery = document.getElementById("gallery");
//     const scrollAmount = 300; // pixels to scroll
//     gallery.scrollBy({ left: direction * scrollAmount, behavior: 'smooth' });
// }
function scrollGallery(direction) {
  const wrapper = document.getElementById("scroll-wrapper");
  const scrollAmount = wrapper.offsetWidth; // scroll by one visible block (3x3)
  wrapper.scrollBy({
    left: direction * scrollAmount,
    behavior: 'smooth'
  });
}

fetch(`/select_query/${filename}`, {
    method: 'POST'
}).then(response => {
    if (response.redirected) {
        window.location.href = response.url;
    }
});

function startWebSocketStream() {
  const socket = new WebSocket(`ws://${location.host}/ws/match_feed`);
  const videoElement = document.getElementById("live-video");

  socket.onopen = () => {
    console.log("📡 WebSocket connected");
  };

  socket.onmessage = (event) => {
    const b64 = event.data;
    videoElement.src = `data:image/jpeg;base64,${b64}`;
  };

  socket.onclose = () => {
    console.log("🔌 WebSocket disconnected");
  };

  socket.onerror = (err) => {
    console.error("🚨 WebSocket error:", err);
  };
}

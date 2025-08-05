// async function startStream() {
//     showSpinner("Starting detection...");
//     const response = await fetch('/start_stream', {
//         method: 'POST'
//     });

//     if (response.ok) {
//         await loadGallery();
//         hideSpinner();
//         alert("Detection complete! Select a person to match.");
//     } else {
//         hideSpinner();
//         alert("Failed to start detection.");
//     }
// }


// async function loadGallery() {
//   const res = await fetch('/get_query_images');
//   const images = await res.json();
//   const gallery = document.getElementById("gallery");
//   gallery.innerHTML = "";

//   const perPage = 9;
//   const totalPages = Math.ceil(images.length / perPage);

//   for (let page = 0; page < totalPages; page++) {
//     const pageContainer = document.createElement("div");
//     pageContainer.className = "gallery-page";
//     pageContainer.style.display = "grid";
//     pageContainer.style.gridTemplateColumns = "repeat(3, 1fr)";
//     pageContainer.style.gridGap = "12px";
//     pageContainer.style.minWidth = "360px"; // width of 3 images + gaps
//     pageContainer.style.flex = "0 0 auto";  // keep fixed page width

//     const pageImages = images.slice(page * perPage, (page + 1) * perPage);

//     pageImages.forEach(filename => {
//       const img = document.createElement("img");
//       img.src = `/query_images/${filename}`;
//       img.style.width = "100px";
//       img.style.height = "140px";
//       img.style.objectFit = "cover";
//       img.style.border = "2px solid transparent";
//       img.style.borderRadius = "6px";
//       img.style.cursor = "pointer";
//       img.style.transition = "transform 0.2s ease";

//       img.onclick = () => {
//   selectImage(filename, img);
// };

//       img.onmouseenter = () => img.style.transform = "scale(1.05)";
//       img.onmouseleave = () => img.style.transform = "scale(1)";

//       pageContainer.appendChild(img);
//     });

//     gallery.appendChild(pageContainer);
//   }
// }


// async function selectImage(filename, imgElement) {
//   await fetch(`/select_query/${filename}`, { method: 'POST' });

//   document.querySelectorAll("#gallery img").forEach(img => {
//     img.style.border = "2px solid transparent";
//   });
//   imgElement.style.border = "3px solid #3498db";

//   const preview = document.getElementById("selected-preview");
//   preview.innerHTML = `
//     <h3>Selected Person</h3>
//     <img src="/query/query.jpg?ts=${Date.now()}" width="150" style="border-radius: 8px;" />
//   `;

//   // Start matching and WebSocket stream
//   await fetch('/start_matching');
//   startWebSocketStream();
// }





// async function resetSession() {
//     showSpinner("Resetting session...");
//     const res = await fetch('/reset', { method: 'POST' });
//     if (res.ok) {
//         document.getElementById("gallery").innerHTML = "";
//         document.getElementById("selected-preview").innerHTML = "";
//         hideSpinner();
//         alert("Session reset.");
//     } else {
//         hideSpinner();
//         alert("Failed to reset.");
//     }
// }

// function showSpinner(msg = "Loading...") {
//     const spinner = document.getElementById("spinner");
//     spinner.style.display = "block";
//     spinner.querySelector("br").nextSibling.textContent = msg;
// }

// function hideSpinner() {
//     document.getElementById("spinner").style.display = "none";
// }


// // function scrollGallery(direction) {
// //     const gallery = document.getElementById("gallery");
// //     const scrollAmount = 300; // pixels to scroll
// //     gallery.scrollBy({ left: direction * scrollAmount, behavior: 'smooth' });
// // }
// function scrollGallery(direction) {
//   const wrapper = document.getElementById("scroll-wrapper");
//   const scrollAmount = wrapper.offsetWidth; // scroll by one visible block (3x3)
//   wrapper.scrollBy({
//     left: direction * scrollAmount,
//     behavior: 'smooth'
//   });
// }

// fetch(`/select_query/${filename}`, {
//     method: 'POST'
// }).then(response => {
//     if (response.redirected) {
//         window.location.href = response.url;
//     }
// });

// // function startWebSocketStream() {
// //   const socket = new WebSocket(`ws://${location.host}/ws/match_feed`);
// //   const videoElement = document.getElementById("live-video");

// //   socket.onopen = () => {
// //     console.log("📡 WebSocket connected");
// //   };

// //   socket.onmessage = (event) => {
// //     const b64 = event.data;
// //     videoElement.src = `data:image/jpeg;base64,${b64}`;
// //   };

// //   socket.onclose = () => {
// //     console.log("🔌 WebSocket disconnected");
// //   };

// //   socket.onerror = (err) => {
// //     console.error("🚨 WebSocket error:", err);
// //   };
// // }

let isMatching = false;
let selectedQueryImage = null;

async function startStream() {
    showSpinner("Starting detection...");
    updateStatus("Detecting persons...");
    
    try {
        const response = await fetch('/start_stream', {
            method: 'POST'
        });
        
        const data = await response.json();

        if (data.status === 'done') {
            console.log("Detection completed:", data);
            await loadGallery();
            hideSpinner();
            updateStatus(`Detection complete! Found ${data.person_count || 'some'} person crops. Select one to match.`);
            
            // Reset camera feeds after detection
            setTimeout(resetCameraFeeds, 1000);
            
            alert(`Detection complete! Found ${data.person_count || 'some'} person crops. Select a person to match.`);
        } else {
            hideSpinner();
            updateStatus(`Detection failed: ${data.message || 'Unknown error'}`);
            alert(`Failed to start detection: ${data.message || 'Unknown error'}`);
        }
    } catch (error) {
        hideSpinner();
        updateStatus("Detection failed!");
        console.error("Detection error:", error);
        alert("Failed to start detection: " + error.message);
    }
}

async function resetCameraFeeds() {
    console.log("Resetting camera feeds...");
    try {
        const response = await fetch('/reset_cameras', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log("Cameras reset successfully");
            // Refresh the camera feeds
            refreshCameraFeeds();
        } else {
            console.error("Failed to reset cameras:", data.message);
        }
    } catch (error) {
        console.error("Error resetting cameras:", error);
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
    showSpinner("Selecting query person...");
    
    const response = await fetch(`/select_query/${filename}`, { method: 'POST' });
    
    if (response.ok) {
        // Update visual selection
        document.querySelectorAll("#gallery img").forEach(img => {
            img.style.border = "2px solid transparent";
        });
        imgElement.style.border = "3px solid #3498db";

        // Show selected preview
        const preview = document.getElementById("selected-preview");
        preview.innerHTML = `
            <h3>Selected Person</h3>
            <img src="/query/query.jpg?ts=${Date.now()}" width="150" style="border-radius: 8px;" />
            <p style="text-align: center; margin-top: 10px;">
                <button class="btn btn-success" onclick="startMatchingProcess()">🎯 Start Matching</button>
            </p>
        `;

        selectedQueryImage = filename;
        updateStatus(`Query person selected: ${filename}`);
        hideSpinner();
        
        // Show the start matching button
        document.getElementById('startMatchingBtn').style.display = 'inline-block';
    } else {
        hideSpinner();
        updateStatus("Failed to select query person");
        alert("Failed to select query person");
    }
}

async function startMatchingProcess() {
    if (!selectedQueryImage) {
        alert("Please select a query person first!");
        return;
    }

    showSpinner("Starting matching system...");
    updateStatus("Initializing matching...");

    try {
        const response = await fetch('/start_matching');
        const data = await response.json();
        
        console.log("Start matching response:", data);
        
        if (data.status === 'matching_started') {
            isMatching = true;
            updateStatus("🎯 Matching Active", true);
            
            // Show the matching section
            document.getElementById('matchingSection').classList.add('active');
            
            // Force refresh the matching feed to show results
            refreshMatchingFeed();
            
            hideSpinner();
            
            // Start checking matching status
            checkMatchingStatus();
            
            // Scroll to matching section
            document.getElementById('matchingSection').scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
            
        } else {
            hideSpinner();
            updateStatus("Failed to start matching");
            alert("Failed to start matching");
        }
    } catch (error) {
        hideSpinner();
        updateStatus("Error starting matching");
        console.error("Error starting matching:", error);
        alert("Error starting matching: " + error.message);
    }
}

async function checkMatchingStatus() {
    try {
        const response = await fetch('/matching_status');
        const status = await response.json();
        console.log("Matching status:", status);
        
        if (!status.matching_initialized && isMatching) {
            updateStatus("⚠️ Matching system not fully initialized");
        }
        
        if (!status.query_exists && isMatching) {
            updateStatus("⚠️ Query image not found");
        }
        
    } catch (error) {
        console.error("Error checking matching status:", error);
    }
    
    // Check again in 5 seconds if matching is active
    if (isMatching) {
        setTimeout(checkMatchingStatus, 5000);
    }
}

async function stopMatching() {
    showSpinner("Stopping matching...");
    
    try {
        const response = await fetch('/stop_matching', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'matching_stopped') {
            isMatching = false;
            updateStatus("Matching stopped");
            
            // Hide the matching section
            document.getElementById('matchingSection').classList.remove('active');
            
            hideSpinner();
        } else {
            hideSpinner();
            updateStatus("Failed to stop matching");
            alert("Failed to stop matching");
        }
    } catch (error) {
        hideSpinner();
        updateStatus("Error stopping matching");
        console.error("Error stopping matching:", error);
        alert("Error stopping matching: " + error.message);
    }
}

async function resetSession() {
    showSpinner("Resetting session...");
    updateStatus("Resetting...");
    
    const res = await fetch('/reset', { method: 'POST' });
    if (res.ok) {
        // Reset UI elements
        document.getElementById("gallery").innerHTML = "";
        document.getElementById("selected-preview").innerHTML = "";
        document.getElementById('matchingSection').classList.remove('active');
        document.getElementById('startMatchingBtn').style.display = 'none';
        
        // Reset variables
        isMatching = false;
        selectedQueryImage = null;
        
        // Refresh camera feeds
        refreshCameraFeeds();
        
        updateStatus("Session reset successfully");
        hideSpinner();
        alert("Session reset.");
    } else {
        hideSpinner();
        updateStatus("Reset failed");
        alert("Failed to reset.");
    }
}

function refreshMatchingFeed() {
    if (isMatching) {
        const matchingFeed = document.getElementById('matchingFeed');
        const timestamp = new Date().getTime();
        matchingFeed.src = `/video_feed?t=${timestamp}`;
    }
}

function refreshCameraFeeds() {
    const timestamp = new Date().getTime();
    document.getElementById('cam0Feed').src = `/video_feed/0?t=${timestamp}`;
    document.getElementById('cam1Feed').src = `/video_feed/1?t=${timestamp}`;
    document.getElementById('matchingFeed').src = `/video_feed?t=${timestamp}`;
}

function updateStatus(message, isActive = false) {
    const statusIndicator = document.getElementById('statusIndicator');
    statusIndicator.textContent = message;
    
    if (isActive) {
        statusIndicator.classList.add('matching');
    } else {
        statusIndicator.classList.remove('matching');
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

function scrollGallery(direction) {
    const wrapper = document.getElementById("scroll-wrapper");
    const scrollAmount = wrapper.offsetWidth; // scroll by one visible block (3x3)
    wrapper.scrollBy({
        left: direction * scrollAmount,
        behavior: 'smooth'
    });
}

// Auto-refresh matching feed every 100ms when matching is active
setInterval(() => {
    if (isMatching) {
        refreshMatchingFeed();
    }
}, 100);

// Auto-refresh camera feeds every 30 seconds to prevent stale connections
setInterval(() => {
    if (!isMatching) {
        refreshCameraFeeds();
    }
}, 30000);

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    updateStatus("Ready - Click 'Start Detection' to begin");
});
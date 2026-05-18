// Enhanced Emergency Medical Locator 
// Based on Google Maps Platform sample code, optimized for mobile app integration
// Uses the new Places API (New) and Routes API for better performance

class EmergencyMedicalLocator {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.userLocation = null;
        this.userMarker = null;
        this.placeMarkers = [];
        this.routePolyline = null;
        this.map = null;
        this.isInitialized = false;
    }

    // Initialize the emergency locator with Google Maps
    async initializeEmergencyLocator(mapElementId = 'map') {
        try {
            // Load required Google Maps libraries
            const { Map, Polyline } = await google.maps.importLibrary("maps");
            const { AdvancedMarkerElement, PinElement } = await google.maps.importLibrary("marker");
            const { Place } = await google.maps.importLibrary("places");
            await google.maps.importLibrary("geometry");

            // Initialize map with emergency-optimized settings
            this.map = new Map(document.getElementById(mapElementId), {
                center: { lat: 3.139, lng: 101.6869 }, // Default to KL
                zoom: 12,
                mapId: 'EMERGENCY_MAP',
                disableDefaultUI: true,
                zoomControl: true,
                // Emergency-friendly styling
                styles: [
                    {
                        featureType: 'poi.medical',
                        elementType: 'all',
                        stylers: [{ visibility: 'on' }, { color: '#ff0000' }]
                    }
                ]
            });

            // Initialize route polyline for directions
            this.routePolyline = new Polyline({
                strokeColor: '#ff0000', // Red for emergency
                strokeOpacity: 0.9,
                strokeWeight: 8,
                map: this.map,
            });

            this.isInitialized = true;
            return { success: true };

        } catch (error) {
            console.error('Failed to initialize emergency locator:', error);
            return { success: false, error: error.message };
        }
    }

    // Get user's current location with high accuracy
    async getCurrentLocation(highAccuracy = true) {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject(new Error('Geolocation not supported'));
                return;
            }

            const options = {
                enableHighAccuracy: highAccuracy,
                timeout: highAccuracy ? 15000 : 10000, // Longer timeout for high accuracy
                maximumAge: 30000 // 30 seconds cache for emergency situations
            };

            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const location = {
                        lat: position.coords.latitude,
                        lng: position.coords.longitude,
                        accuracy: position.coords.accuracy
                    };
                    this.userLocation = location;
                    resolve(location);
                },
                (error) => {
                    // Try again with lower accuracy if high accuracy fails
                    if (highAccuracy && error.code === error.TIMEOUT) {
                        this.getCurrentLocation(false).then(resolve).catch(reject);
                    } else {
                        reject(new Error(`Location error: ${error.message}`));
                    }
                },
                options
            );
        });
    }

    // Main emergency function - Find medical facilities using new Places API
    async findEmergencyMedicalHelp(options = {}) {
        try {
            const {
                radius = 5000, // 5km for emergency
                maxResults = 10,
                includePharmacies = true,
                showOnMap = true
            } = options;

            // Get user location if not already available
            if (!this.userLocation) {
                await this.getCurrentLocation();
            }

            // Center map on user location if map is available
            if (this.map && showOnMap) {
                this.map.setCenter(this.userLocation);
                this.map.setZoom(14);
                await this.addUserMarker();
            }

            // Search for medical facilities using new Places API
            const facilities = await this.searchMedicalFacilities(radius, maxResults, includePharmacies);

            if (showOnMap && facilities.length > 0) {
                await this.displayFacilitiesOnMap(facilities);
            }

            // Calculate distances and sort by proximity
            const facilitiesWithDistance = this.calculateDistancesAndSort(facilities);

            return {
                success: true,
                userLocation: this.userLocation,
                facilities: facilitiesWithDistance.map(facility => ({
                    id: facility.id,
                    name: facility.displayName,
                    address: facility.formattedAddress,
                    location: {
                        lat: facility.location.lat,
                        lng: facility.location.lng
                    },
                    distance: facility.distance,
                    distanceText: facility.distanceText,
                    types: facility.types,
                    rating: facility.rating || null,
                    isOpen: facility.regularOpeningHours ? 
                           this.getCurrentOpenStatus(facility.regularOpeningHours) : null,
                    phone: facility.nationalPhoneNumber || null,
                    // Emergency-specific features
                    emergencyServices: this.detectEmergencyServices(facility),
                    // Ready-to-use navigation
                    directionsUrl: this.getGoogleMapsDirectionsUrl(facility.location),
                    // For quick route calculation
                    canCalculateRoute: true
                }))
            };

        } catch (error) {
            console.error('Emergency search failed:', error);
            return {
                success: false,
                error: error.message,
                emergencyContacts: this.getLocalEmergencyContacts()
            };
        }
    }

    // Search medical facilities using the new Places API
    async searchMedicalFacilities(radius, maxResults, includePharmacies) {
        const { Place } = await google.maps.importLibrary("places");
        
        const includedTypes = ['hospital', 'doctor'];
        if (includePharmacies) {
            includedTypes.push('pharmacy');
        }

        const request = {
            locationRestriction: {
                center: this.userLocation,
                radius: radius,
            },
            includedTypes: includedTypes,
            fields: [
                'displayName', 
                'location', 
                'formattedAddress',
                'types',
                'rating',
                'regularOpeningHours',
                'nationalPhoneNumber',
                'id'
            ],
            maxResultCount: maxResults,
            // Prioritize places that are currently open
            rankPreference: 'RELEVANCE'
        };

        try {
            const { places } = await Place.searchNearby(request);
            return places || [];
        } catch (error) {
            console.error('Places search failed:', error);
            return [];
        }
    }

    // Add user location marker to map
    async addUserMarker() {
        if (!this.map) return;

        const { AdvancedMarkerElement, PinElement } = await google.maps.importLibrary("marker");
        
        const userPin = new PinElement({
            background: '#4285f4',
            borderColor: '#1a73e8',
            glyphColor: '#ffffff',
            glyph: '📍',
            scale: 1.2
        });

        this.userMarker = new AdvancedMarkerElement({
            map: this.map,
            position: this.userLocation,
            content: userPin.element,
            title: 'Your Location',
        });
    }

    // Display facilities on map with emergency-styled markers
    async displayFacilitiesOnMap(facilities) {
        if (!this.map) return;

        const { AdvancedMarkerElement, PinElement } = await google.maps.importLibrary("marker");
        
        // Clear existing markers
        this.clearPlaceMarkers();

        facilities.forEach(place => {
            const isHospital = place.types.includes('hospital');
            const isPharmacy = place.types.includes('pharmacy');
            
            const hospitalPin = new PinElement({
                background: isHospital ? '#dc3545' : isPharmacy ? '#28a745' : '#fd7e14',
                borderColor: isHospital ? '#a02834' : isPharmacy ? '#1e7e34' : '#d76e05',
                glyphColor: '#ffffff',
                glyph: isHospital ? '🏥' : isPharmacy ? '💊' : '🩺',
                scale: isHospital ? 1.3 : 1.0
            });

            const marker = new AdvancedMarkerElement({
                map: this.map,
                position: place.location,
                content: hospitalPin.element,
                title: place.displayName,
            });

            // Add click listener for quick info
            marker.addListener('click', () => {
                this.showFacilityInfo(place, marker);
            });

            this.placeMarkers.push(marker);
        });
    }

    // Calculate and display route to selected facility
    async calculateEmergencyRoute(destination, options = {}) {
        if (!this.userLocation) {
            throw new Error('User location not available');
        }

        const {
            travelMode = 'DRIVE',
            showOnMap = true
        } = options;

        const request = {
            origin: { location: this.userLocation },
            destination: { location: destination },
            travelMode: travelMode,
            // Request emergency-relevant route info
            computeAlternativeRoutes: false,
            routeModifiers: {
                avoidTolls: false,
                avoidHighways: false,
                avoidFerries: true
            }
        };

        try {
            const response = await fetch("https://routes.googleapis.com/directions/v2:computeRoutes", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": this.apiKey,
                    "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,routes.legs.steps.navigationInstruction"
                },
                body: JSON.stringify(request)
            });

            if (!response.ok) {
                throw new Error(`Route calculation failed: ${response.status}`);
            }

            const data = await response.json();

            if (data.routes && data.routes.length > 0) {
                const route = data.routes[0];
                
                if (showOnMap && this.map && this.routePolyline) {
                    // Display route on map
                    const decodedPath = google.maps.geometry.encoding.decodePath(
                        route.polyline.encodedPolyline
                    );
                    this.routePolyline.setPath(decodedPath);
                    
                    // Fit map to show entire route
                    const bounds = new google.maps.LatLngBounds();
                    decodedPath.forEach(point => bounds.extend(point));
                    this.map.fitBounds(bounds);
                }

                const distance = (route.distanceMeters / 1000).toFixed(2);
                const durationSeconds = parseInt(route.duration.slice(0, -1));
                const durationMinutes = Math.round(durationSeconds / 60);

                return {
                    success: true,
                    distance: `${distance} km`,
                    duration: `${durationMinutes} min`,
                    durationSeconds: durationSeconds,
                    // For navigation apps
                    encodedPolyline: route.polyline.encodedPolyline,
                    // Basic navigation instructions
                    instructions: this.extractBasicInstructions(route)
                };
            } else {
                throw new Error('No route found');
            }

        } catch (error) {
            console.error('Route calculation failed:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    // Quick emergency search - optimized for one-tap emergency button
    async quickEmergencySearch() {
        try {
            const result = await this.findEmergencyMedicalHelp({
                radius: 3000, // 3km for urgent cases
                maxResults: 5,
                showOnMap: false // Skip map for faster results
            });

            if (result.success && result.facilities.length > 0) {
                const closest = result.facilities[0];
                
                // Calculate route to closest facility
                const route = await this.calculateEmergencyRoute(closest.location, {
                    showOnMap: false
                });

                return {
                    success: true,
                    userLocation: result.userLocation,
                    closestFacility: {
                        ...closest,
                        route: route
                    },
                    allFacilities: result.facilities,
                    // Quick actions for emergency
                    quickActions: {
                        navigate: closest.directionsUrl,
                        call: closest.phone,
                        emergency: this.getLocalEmergencyContacts().ambulance
                    }
                };
            } else {
                return {
                    success: false,
                    error: 'No nearby medical facilities found',
                    emergencyContacts: this.getLocalEmergencyContacts()
                };
            }

        } catch (error) {
            return {
                success: false,
                error: error.message,
                emergencyContacts: this.getLocalEmergencyContacts()
            };
        }
    }

    // Utility functions
    calculateDistancesAndSort(facilities) {
        return facilities
            .map(facility => {
                const distance = this.calculateDistance(
                    this.userLocation.lat,
                    this.userLocation.lng,
                    facility.location.lat,
                    facility.location.lng
                );
                return {
                    ...facility,
                    distance: distance,
                    distanceText: distance < 1 ? 
                        `${Math.round(distance * 1000)}m` : 
                        `${distance.toFixed(1)}km`
                };
            })
            .sort((a, b) => a.distance - b.distance);
    }

    calculateDistance(lat1, lon1, lat2, lon2) {
        const R = 6371; // Earth's radius in km
        const dLat = this.toRad(lat2 - lat1);
        const dLon = this.toRad(lon2 - lon1);
        const a = 
            Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(this.toRad(lat1)) * Math.cos(this.toRad(lat2)) *
            Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    toRad(deg) {
        return deg * (Math.PI/180);
    }

    detectEmergencyServices(facility) {
        const name = facility.displayName.toLowerCase();
        const types = facility.types || [];
        
        return {
            hasER: types.includes('hospital') || name.includes('emergency'),
            is24Hours: this.is24HoursFacility(facility.regularOpeningHours),
            hasAmbulance: types.includes('hospital'),
            type: types.includes('hospital') ? 'Hospital' : 
                  types.includes('pharmacy') ? 'Pharmacy' : 'Clinic'
        };
    }

    getCurrentOpenStatus(openingHours) {
        if (!openingHours || !openingHours.periods) return null;
        
        const now = new Date();
        const dayOfWeek = now.getDay(); // 0 = Sunday, 1 = Monday, etc.
        const currentTime = now.getHours() * 100 + now.getMinutes();
        
        // Check if open today
        const todayHours = openingHours.periods.find(period => 
            period.open && period.open.day === dayOfWeek
        );
        
        if (!todayHours || !todayHours.open) return false;
        if (!todayHours.close) return true; // 24 hours
        
        const openTime = parseInt(todayHours.open.time);
        const closeTime = parseInt(todayHours.close.time);
        
        return currentTime >= openTime && currentTime <= closeTime;
    }

    is24HoursFacility(openingHours) {
        if (!openingHours || !openingHours.periods) return false;
        
        return openingHours.periods.some(period => 
            period.open && !period.close
        );
    }

    getGoogleMapsDirectionsUrl(destination) {
        return `https://www.google.com/maps/dir/?api=1&destination=${destination.lat},${destination.lng}&travelmode=driving`;
    }

    getLocalEmergencyContacts() {
        return {
            ambulance: '999', // Malaysia
            police: '999',
            fire: '999',
            medical: '15454' // Malaysia medical emergency
        };
    }

    extractBasicInstructions(route) {
        // Extract basic turn-by-turn instructions for simple navigation
        if (!route.legs || !route.legs[0] || !route.legs[0].steps) {
            return [];
        }
        
        return route.legs[0].steps
            .filter(step => step.navigationInstruction)
            .map(step => step.navigationInstruction.instructions)
            .slice(0, 5); // First 5 instructions for emergency
    }

    showFacilityInfo(facility, marker) {
        // For integration with your app's info window/modal
        const distance = this.calculateDistance(
            this.userLocation.lat,
            this.userLocation.lng,
            facility.location.lat,
            facility.location.lng
        );
        
        return {
            name: facility.displayName,
            address: facility.formattedAddress,
            distance: distance < 1 ? `${Math.round(distance * 1000)}m` : `${distance.toFixed(1)}km`,
            phone: facility.nationalPhoneNumber,
            isOpen: this.getCurrentOpenStatus(facility.regularOpeningHours),
            emergencyServices: this.detectEmergencyServices(facility)
        };
    }

    clearPlaceMarkers() {
        this.placeMarkers.forEach(marker => {
            marker.map = null;
        });
        this.placeMarkers = [];
    }

    clearRoute() {
        if (this.routePolyline) {
            this.routePolyline.setPath([]);
        }
    }
}

// USAGE EXAMPLE - Integration with Emergency Button
const emergencyLocator = new EmergencyMedicalLocator('YOUR_API_KEY');

// Initialize (call this when your app starts)
async function initializeEmergencyFeature() {
    const result = await emergencyLocator.initializeEmergencyLocator();
    if (!result.success) {
        console.error('Failed to initialize emergency feature:', result.error);
    }
}

// Emergency button handler - ONE-TAP EMERGENCY SEARCH
async function handleEmergencyButtonPress() {
    try {
        // Show loading indicator
        showEmergencyLoading('🚨 Finding nearest medical help...');
        
        // Quick emergency search
        const result = await emergencyLocator.quickEmergencySearch();
        
        hideEmergencyLoading();
        
        if (result.success) {
            const closest = result.closestFacility;
            
            // Show emergency results with quick action buttons
            showEmergencyResults({
                facility: closest,
                distance: closest.distanceText,
                route: closest.route,
                actions: {
                    navigate: () => window.open(result.quickActions.navigate, '_blank'),
                    call: result.quickActions.call ? () => window.open(`tel:${result.quickActions.call}`) : null,
                    emergency: () => window.open(`tel:${result.quickActions.emergency}`)
                },
                allFacilities: result.allFacilities
            });
            
        } else {
            // Show emergency contacts as fallback
            showEmergencyFallback(result.emergencyContacts);
        }
        
    } catch (error) {
        hideEmergencyLoading();
        console.error('Emergency search failed:', error);
        showEmergencyContacts();
    }
}

// Helper functions - customize these for your app's UI
function showEmergencyLoading(message) {
    console.log(message);
    // Implement your emergency loading UI
}

function hideEmergencyLoading() {
    // Hide loading UI
}

function showEmergencyResults(data) {
    console.log('Closest facility:', data.facility.name);
    console.log('Distance:', data.distance);
    if (data.route.success) {
        console.log('Route:', data.route.distance, data.route.duration);
    }
    // Implement emergency results UI with action buttons
}

function showEmergencyFallback(contacts) {
    console.log('Emergency contacts:', contacts);
    // Show emergency numbers when location fails
}

function showEmergencyContacts() {
    console.log('Call 999 for emergency services');
    // Last resort emergency contacts
}

// For React Native/Ionic/Cordova integration
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        EmergencyMedicalLocator,
        handleEmergencyButtonPress,
        initializeEmergencyFeature
    };
}
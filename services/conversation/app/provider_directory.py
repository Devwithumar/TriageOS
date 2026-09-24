import json
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class ProviderDirectoryError(RuntimeError):
    pass


class ProviderDirectoryTimeout(ProviderDirectoryError):
    pass


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_until: float = 0.0
    last_error: str | None = None


@dataclass(frozen=True)
class Provider:
    provider_id: str
    name: str
    address: str | None
    category: str
    latitude: float
    longitude: float
    distance_km: float
    phone: str | None = None
    website: str | None = None
    source: str = "OpenStreetMap"

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "address": self.address,
            "category": self.category,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "distance_km": round(self.distance_km, 2),
            "phone": self.phone,
            "website": self.website,
            "source": self.source,
        }


@dataclass(frozen=True)
class ProviderSearchResult:
    location: str
    providers: list[Provider]
    source: str = "OpenStreetMap"
    error: str | None = None
    error_code: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "location": self.location,
            "providers": [provider.as_dict() for provider in self.providers],
            "source": self.source,
            "error": self.error,
            "error_code": self.error_code,
        }


class ProviderDirectory:
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._config = self._load_config()
        self._provider = os.getenv("PROVIDER_DIRECTORY_PROVIDER", "osm").strip().lower()
        self._cache: dict[tuple[str, str], tuple[float, ProviderSearchResult]] = {}
        self._circuits: dict[str, _CircuitState] = {}
        self._http_client_factory = httpx.Client
        self._lock = threading.Lock()
        self._last_nominatim_request = 0.0
        self._user_agent = os.getenv(
            "OSM_USER_AGENT",
            "TriageOS/0.1 (local development; provider directory contact not configured)",
        )
        self._nominatim_url = os.getenv(
            "NOMINATIM_URL", "https://nominatim.openstreetmap.org/search"
        )
        self._overpass_url = os.getenv(
            "OVERPASS_URL", "https://overpass-api.de/api/interpreter"
        )
        self._overpass_fallback_url = os.getenv(
            "OVERPASS_FALLBACK_URL", "https://overpass.kumi.systems/api/interpreter"
        )
        self._cache_ttl_seconds = int(os.getenv("PROVIDER_DIRECTORY_CACHE_TTL", "600"))
        self._radius_meters = int(os.getenv("PROVIDER_DIRECTORY_RADIUS_METERS", "5000"))
        self._search_timeout_seconds = float(os.getenv("PROVIDER_DIRECTORY_SEARCH_TIMEOUT", "8"))
        self._overpass_max_attempts = max(
            1, int(os.getenv("PROVIDER_DIRECTORY_OVERPASS_MAX_ATTEMPTS", "1"))
        )
        self._operation_timeout_seconds = float(
            os.getenv("PROVIDER_DIRECTORY_OPERATION_TIMEOUT", "20")
        )
        self._nominatim_timeout_seconds = float(
            os.getenv("PROVIDER_DIRECTORY_GEOCODE_TIMEOUT", "5")
        )
        self._max_attempts = max(1, int(os.getenv("PROVIDER_DIRECTORY_MAX_ATTEMPTS", "2")))
        self._retry_backoff_seconds = max(
            0.0, float(os.getenv("PROVIDER_DIRECTORY_RETRY_BACKOFF", "0.25"))
        )
        self._circuit_failure_threshold = max(
            1, int(os.getenv("PROVIDER_DIRECTORY_CIRCUIT_FAILURE_THRESHOLD", "3"))
        )
        self._circuit_open_seconds = max(
            1.0, float(os.getenv("PROVIDER_DIRECTORY_CIRCUIT_OPEN_SECONDS", "30"))
        )

    def search(self, care_setting: str, location: str, reason: str | None = None) -> ProviderSearchResult:
        if self._provider != "osm":
            return ProviderSearchResult(
                location=location,
                providers=[],
                error=f"provider directory provider '{self._provider}' is not enabled",
                error_code="misconfigured",
            )
        profile_name, profile = self._resolve_profile(care_setting)
        cache_key = (profile_name, " ".join(location.lower().split()))
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < self._cache_ttl_seconds:
            return cached[1]

        deadline = time.monotonic() + self._operation_timeout_seconds
        try:
            latitude, longitude, resolved_location = self._geocode(location, deadline)
            directory_error: Exception | None = None
            providers = []
            used_nominatim = False
            try:
                providers = self._search_nominatim_providers(
                    care_setting=care_setting,
                    location=location,
                    profile_name=profile_name,
                    profile=profile,
                    origin_latitude=latitude,
                    origin_longitude=longitude,
                    deadline=deadline,
                )
                used_nominatim = bool(providers)
            except (httpx.HTTPError, ProviderDirectoryError) as exc:
                directory_error = exc

            if not providers:
                try:
                    elements = self._search_openstreetmap(
                        latitude,
                        longitude,
                        profile["filters"],
                        profile.get("name_pattern"),
                        deadline,
                    )
                    providers = self._providers_from_elements(
                        elements,
                        profile_name,
                        latitude,
                        longitude,
                    )
                    if providers:
                        directory_error = None
                except (httpx.HTTPError, ProviderDirectoryError) as exc:
                    directory_error = exc

            result = ProviderSearchResult(
                location=location,
                providers=providers,
                source="OpenStreetMap/Nominatim" if used_nominatim else "OpenStreetMap",
                error=(
                    "The provider directory search is temporarily unavailable."
                    if directory_error is not None and not providers
                    else None
                ),
                error_code=(
                    _provider_error_code(directory_error)
                    if directory_error is not None and not providers
                    else None
                ),
            )
        except (httpx.HTTPError, ValueError, KeyError, ProviderDirectoryError) as exc:
            result = ProviderSearchResult(
                location=location,
                providers=[],
                error=str(exc),
                error_code=_provider_error_code(exc),
            )

        if result.error is None:
            self._cache[cache_key] = (time.monotonic(), result)
        return result

    def search_named_provider(self, provider_name: str, location: str) -> ProviderSearchResult:
        cache_key = (f"named:{provider_name.lower().strip()}", " ".join(location.lower().split()))
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < self._cache_ttl_seconds:
            return cached[1]
        deadline = time.monotonic() + self._operation_timeout_seconds
        try:
            latitude, longitude, resolved_location = self._geocode(location, deadline)
            directory_error: Exception | None = None
            providers = []
            used_nominatim = False
            try:
                providers = self._search_nominatim_named_provider(
                    provider_name=provider_name,
                    location=location,
                    origin_latitude=latitude,
                    origin_longitude=longitude,
                    deadline=deadline,
                )
                used_nominatim = bool(providers)
            except (httpx.HTTPError, ProviderDirectoryError) as exc:
                directory_error = exc

            if not providers:
                try:
                    elements = self._search_openstreetmap(
                        latitude,
                        longitude,
                        [],
                        re.escape(provider_name.strip()),
                        deadline,
                    )
                    providers = self._providers_from_elements(
                        elements,
                        "named_provider",
                        latitude,
                        longitude,
                    )
                    if providers:
                        directory_error = None
                except (httpx.HTTPError, ProviderDirectoryError) as exc:
                    directory_error = exc
            result = ProviderSearchResult(
                location=location,
                providers=providers,
                source="OpenStreetMap/Nominatim" if used_nominatim else "OpenStreetMap",
                error=(
                    "The provider directory search is temporarily unavailable."
                    if directory_error is not None and not providers
                    else None
                ),
                error_code=(
                    _provider_error_code(directory_error)
                    if directory_error is not None and not providers
                    else None
                ),
            )
        except (httpx.HTTPError, ValueError, KeyError, ProviderDirectoryError) as exc:
            result = ProviderSearchResult(
                location=location,
                providers=[],
                error=str(exc),
                error_code=_provider_error_code(exc),
            )
        if result.error is None:
            self._cache[cache_key] = (time.monotonic(), result)
        return result

    def _geocode(self, location: str, deadline: float | None = None) -> tuple[float, float, str]:
        params = {
            "q": location,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
        }
        places = self._nominatim_search(params, deadline)

        if not places:
            raise ProviderDirectoryError(f"I could not locate {location}.")
        place = places[0]
        return float(place["lat"]), float(place["lon"]), str(place.get("display_name", location))

    def _search_nominatim_providers(
        self,
        care_setting: str,
        location: str,
        profile_name: str,
        profile: dict[str, Any],
        origin_latitude: float,
        origin_longitude: float,
        deadline: float | None = None,
    ) -> list[Provider]:
        search_term = _profile_search_term(care_setting, profile_name)
        places = self._nominatim_search(
            {
                "q": f"{search_term}, {location}",
                "format": "jsonv2",
                "limit": 10,
                "addressdetails": 1,
                "namedetails": 1,
                "extratags": 1,
                "layer": "poi",
            },
            deadline,
        )
        return self._providers_from_nominatim(
            places,
            profile_name,
            profile,
            origin_latitude,
            origin_longitude,
        )

    def _search_nominatim_named_provider(
        self,
        provider_name: str,
        location: str,
        origin_latitude: float,
        origin_longitude: float,
        deadline: float | None = None,
    ) -> list[Provider]:
        places = self._nominatim_search(
            {
                "q": f"{provider_name}, {location}",
                "format": "jsonv2",
                "limit": 10,
                "addressdetails": 1,
                "namedetails": 1,
                "extratags": 1,
                "layer": "poi",
            },
            deadline,
        )
        needle = _normalize_search_text(provider_name)
        providers: list[Provider] = []
        for place in places:
            if not _is_usable_poi(place):
                continue
            searchable = _normalize_search_text(
                " ".join(
                    str(value)
                    for value in (
                        place.get("name"),
                        place.get("display_name"),
                        place.get("namedetails", {}).get("name"),
                    )
                    if value
                )
            )
            if needle not in searchable:
                continue
            provider = self._provider_from_nominatim_place(
                place,
                "named_provider",
                origin_latitude,
                origin_longitude,
            )
            if provider and provider.distance_km <= self._radius_meters / 1000:
                providers.append(provider)
        return sorted(providers, key=lambda provider: provider.distance_km)[:5]

    def _nominatim_search(
        self,
        params: dict[str, Any],
        deadline: float | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            now = time.monotonic()
            next_allowed = max(now, self._last_nominatim_request + 1.05)
            wait_seconds = next_allowed - now
            remaining = _remaining_seconds(deadline)
            if remaining is not None and wait_seconds >= remaining:
                raise ProviderDirectoryTimeout("The provider directory operation timed out.")
            self._last_nominatim_request = next_allowed
        if wait_seconds:
            time.sleep(wait_seconds)
        if deadline is not None and _remaining_seconds(deadline) <= 0:
            raise ProviderDirectoryTimeout("The provider directory operation timed out.")

        headers = {"User-Agent": self._user_agent, "Accept-Language": "en"}
        places = self._request_json(
            "GET",
            self._nominatim_url,
            headers=headers,
            params=params,
            timeout_seconds=self._nominatim_timeout_seconds,
            deadline=deadline,
        )
        if not isinstance(places, list):
            raise ProviderDirectoryError("The provider directory returned an invalid response.")
        return places

    def _providers_from_nominatim(
        self,
        places: list[dict[str, Any]],
        category: str,
        profile: dict[str, Any],
        origin_latitude: float,
        origin_longitude: float,
    ) -> list[Provider]:
        providers: list[Provider] = []
        seen: set[str] = set()
        for place in places:
            if not _matches_profile(place, profile):
                continue
            provider = self._provider_from_nominatim_place(
                place,
                category,
                origin_latitude,
                origin_longitude,
            )
            if (
                provider
                and provider.distance_km <= self._radius_meters / 1000
                and provider.provider_id not in seen
            ):
                seen.add(provider.provider_id)
                providers.append(provider)
        return sorted(providers, key=lambda provider: provider.distance_km)[:5]

    @staticmethod
    def _provider_from_nominatim_place(
        place: dict[str, Any],
        category: str,
        origin_latitude: float,
        origin_longitude: float,
    ) -> Provider | None:
        if "lat" not in place or "lon" not in place:
            return None
        tags = place.get("extratags")
        if not isinstance(tags, dict):
            tags = {}
        namedetails = place.get("namedetails")
        if not isinstance(namedetails, dict):
            namedetails = {}
        address = _nominatim_address(place)
        name = str(
            place.get("name")
            or namedetails.get("name")
            or str(place.get("display_name", "")).split(",", 1)[0]
        ).strip()
        if not name:
            return None
        provider_id = f"nominatim:{place.get('osm_type', 'place')}:{place.get('osm_id', name)}"
        latitude = float(place["lat"])
        longitude = float(place["lon"])
        return Provider(
            provider_id=provider_id,
            name=name,
            address=address or str(place.get("display_name", "")) or None,
            category=category,
            latitude=latitude,
            longitude=longitude,
            distance_km=_distance_km(origin_latitude, origin_longitude, latitude, longitude),
            phone=tags.get("phone") or tags.get("contact:phone"),
            website=tags.get("website") or tags.get("contact:website"),
            source="OpenStreetMap/Nominatim",
        )

    def _search_openstreetmap(
        self,
        latitude: float,
        longitude: float,
        filters: list[list[Any]],
        name_pattern: str | None,
        deadline: float | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        for key, values in filters:
            encoded_values = "|".join(str(value).replace('"', '') for value in values)
            clauses.append(
                f'nwr(around:{self._radius_meters},{latitude},{longitude})["{key}"~"^{encoded_values}$"];'
            )
        if name_pattern:
            safe_pattern = name_pattern.replace('"', '')
            clauses.append(
                f'nwr(around:{self._radius_meters},{latitude},{longitude})["name"~"{safe_pattern}",i];'
            )
        query = "[out:json][timeout:20];(" + "".join(clauses) + ");out center tags;"
        headers = {"User-Agent": self._user_agent}
        last_error: Exception | None = None
        urls = [self._overpass_url]
        if self._overpass_fallback_url and self._overpass_fallback_url not in urls:
            urls.append(self._overpass_fallback_url)
        for url in urls:
            try:
                payload = self._request_json(
                    "POST",
                    url,
                    headers=headers,
                    data={"data": query},
                    timeout_seconds=self._search_timeout_seconds,
                    deadline=deadline,
                    max_attempts=self._overpass_max_attempts,
                )
                if not isinstance(payload, dict):
                    raise ProviderDirectoryError("The provider directory returned an invalid response.")
                return payload.get("elements", [])
            except (httpx.HTTPError, ProviderDirectoryError) as exc:
                last_error = exc
        if isinstance(last_error, ProviderDirectoryTimeout):
            raise last_error
        raise ProviderDirectoryError("The provider directory search is temporarily unavailable.") from last_error

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        timeout_seconds: float,
        deadline: float | None,
        max_attempts: int | None = None,
    ) -> Any:
        endpoint = _endpoint_name(url)
        self._check_circuit(endpoint)
        last_error: Exception | None = None
        attempts = max(1, max_attempts or self._max_attempts)
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                self._check_circuit(endpoint)
            remaining = _remaining_seconds(deadline)
            if remaining is not None and remaining <= 0:
                raise ProviderDirectoryTimeout("The provider directory operation timed out.")
            request_timeout = timeout_seconds if remaining is None else min(timeout_seconds, remaining)
            try:
                with self._http_client_factory(
                    timeout=request_timeout,
                    headers=headers,
                    follow_redirects=True,
                ) as client:
                    response = client.request(method, url, params=params, data=data)
                    response.raise_for_status()
                    payload = response.json()
                self._record_success(endpoint)
                return payload
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if not _retryable_status(exc.response.status_code):
                    self._record_failure(endpoint, exc)
                    raise
            except httpx.RequestError as exc:
                last_error = exc
            except ValueError as exc:
                self._record_failure(endpoint, exc)
                raise ProviderDirectoryError("The provider directory returned an invalid response.") from exc

            self._record_failure(endpoint, last_error)
            if attempt >= attempts:
                break
            remaining = _remaining_seconds(deadline)
            backoff = self._retry_backoff_seconds * attempt
            if remaining is not None:
                backoff = min(backoff, max(0.0, remaining))
            if backoff:
                time.sleep(backoff)
        if isinstance(last_error, httpx.TimeoutException):
            raise ProviderDirectoryTimeout("The provider directory operation timed out.") from last_error
        raise ProviderDirectoryError("The provider directory search is temporarily unavailable.") from last_error

    def _check_circuit(self, endpoint: str) -> None:
        now = time.monotonic()
        with self._lock:
            circuit = self._circuits.get(endpoint)
            if circuit is None:
                return
            if circuit.opened_until > now:
                raise ProviderDirectoryError("The provider directory endpoint circuit is open.")
            if circuit.opened_until:
                circuit.opened_until = 0.0
                circuit.consecutive_failures = 0
                circuit.last_error = None

    def _record_success(self, endpoint: str) -> None:
        with self._lock:
            circuit = self._circuits.get(endpoint)
            if circuit:
                circuit.consecutive_failures = 0
                circuit.opened_until = 0.0
                circuit.last_error = None

    def _record_failure(self, endpoint: str, error: Exception | None) -> None:
        now = time.monotonic()
        with self._lock:
            circuit = self._circuits.setdefault(endpoint, _CircuitState())
            circuit.consecutive_failures += 1
            circuit.last_error = type(error).__name__ if error else "unknown"
            if circuit.consecutive_failures >= self._circuit_failure_threshold:
                circuit.opened_until = now + self._circuit_open_seconds
        self._logger.warning(
            "provider directory endpoint failure endpoint=%s failures=%s error=%s",
            endpoint,
            circuit.consecutive_failures,
            circuit.last_error,
        )

    def health(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            endpoints = {
                endpoint: {
                    "status": (
                        "open"
                        if state.opened_until > now
                        else "degraded"
                        if state.consecutive_failures
                        else "closed"
                    ),
                    "consecutive_failures": state.consecutive_failures,
                    "retry_after_seconds": round(max(0.0, state.opened_until - now), 2)
                    if state.opened_until > now
                    else 0.0,
                    "last_error": state.last_error,
                }
                for endpoint, state in self._circuits.items()
            }
        return {
            "provider": self._provider,
            "status": (
                "degraded"
                if any(item["status"] != "closed" for item in endpoints.values())
                else "ready"
            ),
            "endpoints": endpoints,
        }

    @staticmethod
    def _providers_from_elements(
        elements: list[dict[str, Any]],
        category: str,
        origin_latitude: float,
        origin_longitude: float,
    ) -> list[Provider]:
        providers: list[Provider] = []
        seen: set[str] = set()
        for element in elements:
            tags = element.get("tags", {})
            name = str(tags.get("name", "")).strip()
            if not name:
                continue
            latitude = element.get("lat", element.get("center", {}).get("lat"))
            longitude = element.get("lon", element.get("center", {}).get("lon"))
            if latitude is None or longitude is None:
                continue
            provider_id = f"osm:{element.get('type', 'place')}:{element.get('id')}"
            if provider_id in seen:
                continue
            seen.add(provider_id)
            address_parts = [
                " ".join(part for part in [tags.get("addr:housenumber"), tags.get("addr:street")] if part),
                tags.get("addr:city") or tags.get("addr:town") or tags.get("addr:suburb"),
            ]
            address = ", ".join(part for part in address_parts if part) or tags.get("addr:full")
            providers.append(
                Provider(
                    provider_id=provider_id,
                    name=name,
                    address=address,
                    category=category,
                    latitude=float(latitude),
                    longitude=float(longitude),
                    distance_km=_distance_km(
                        origin_latitude,
                        origin_longitude,
                        float(latitude),
                        float(longitude),
                    ),
                    phone=tags.get("phone") or tags.get("contact:phone"),
                    website=tags.get("website") or tags.get("contact:website"),
                )
            )
        return sorted(providers, key=lambda provider: provider.distance_km)[:5]

    def _resolve_profile(self, care_setting: str) -> tuple[str, dict[str, Any]]:
        normalized = " ".join(care_setting.lower().split())
        profiles = self._config["profiles"]
        for name, profile in profiles.items():
            if any(alias in normalized for alias in profile.get("aliases", [])):
                return name, profile
        default_name = self._config["default_profile"]
        return default_name, profiles[default_name]

    @staticmethod
    def _load_config() -> dict[str, Any]:
        path = Path(__file__).resolve().parents[3] / "config" / "provider_directory.json"
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)


def _distance_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    radius = 6371.0
    latitude_delta = math.radians(latitude_b - latitude_a)
    longitude_delta = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(math.radians(latitude_a))
        * math.cos(math.radians(latitude_b))
        * math.sin(longitude_delta / 2) ** 2
    )
    return radius * 2 * math.asin(math.sqrt(haversine))


def _profile_search_term(care_setting: str, profile_name: str) -> str:
    normalized = " ".join(care_setting.lower().split())
    if profile_name == "veterinary":
        return "veterinary clinic"
    if profile_name == "dental":
        return "dentist"
    if any(term in normalized for term in ("hospital", "emergency")):
        return "hospital"
    if any(term in normalized for term in ("doctor", "doc", "gp", "general practitioner")):
        return "doctor"
    if "clinic" in normalized:
        return "clinic"
    return "medical clinic"


def _normalize_search_text(value: str) -> str:
    return " ".join(value.lower().split())


def _matches_profile(place: dict[str, Any], profile: dict[str, Any]) -> bool:
    if not _is_usable_poi(place):
        return False
    name_pattern = profile.get("name_pattern")
    if not name_pattern:
        return True
    namedetails = place.get("namedetails", {})
    searchable = " ".join(
        str(value)
        for value in (
            place.get("name"),
            namedetails.get("name"),
            place.get("class"),
            place.get("type"),
            place.get("category"),
        )
        if value
    )
    return bool(re.search(str(name_pattern), searchable, re.IGNORECASE))


def _is_usable_poi(place: dict[str, Any]) -> bool:
    category = str(place.get("category", "")).lower()
    place_type = str(place.get("type", "")).lower()
    if category in {"amenity", "healthcare", "office"}:
        return True
    return category == "shop" and place_type in {"pharmacy", "chemist"}


def _nominatim_address(place: dict[str, Any]) -> str | None:
    address = place.get("address", {})
    if not isinstance(address, dict):
        return None
    street = " ".join(
        str(part)
        for part in (address.get("house_number"), address.get("road"))
        if part
    )
    locality = next(
        (
            str(address[key])
            for key in ("city", "town", "village", "municipality", "suburb", "state")
            if address.get(key)
        ),
        None,
    )
    parts = [part for part in (street, locality) if part]
    return ", ".join(parts) if parts else None


def _provider_error_code(error: Exception | None) -> str:
    if isinstance(error, (httpx.TimeoutException, ProviderDirectoryTimeout)):
        return "timeout"
    if isinstance(error, ProviderDirectoryError) and str(error).startswith("I could not locate"):
        return "location_not_found"
    if isinstance(error, (ValueError, KeyError)):
        return "invalid_response"
    return "unavailable"


def _remaining_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _endpoint_name(url: str) -> str:
    return re.sub(r"^https?://", "", url).split("/", 1)[0].lower()

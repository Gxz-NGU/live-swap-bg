// Builds the two halves of a Live Photo and inspects them.
//
//   pair <edited.mov> <template-live-photo.mov> <output.mov> <content-identifier>
//       Re-wraps the edited video with the template's timed metadata track (the still-image-time marker that
//       makes Photos treat the movie as a Live Photo) and a new content identifier. Of the template's file-level
//       metadata only the keys a Live Photo needs are kept: the phone model, software version, dates and any
//       location never reach the output.
//   stamp <input.jpg> <content-identifier> <output.jpg>
//       Writes the identifier into the photo's Apple MakerNote (key 17), which pairs it with the movie.
//   inspect <movie.mov>        prints the identifier, duration and still-image-time markers as JSON
//   inspect-photo <photo.jpg>  prints the photo's identifier and size as JSON

import AVFoundation
import Foundation
import ImageIO

struct ToolError: Error, CustomStringConvertible { let description: String }
func fail(_ message: String) throws -> Never { throw ToolError(description: message) }

func printJSON(_ value: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

let keptFileMetadata: Set<AVMetadataIdentifier> = [
    AVMetadataIdentifier(rawValue: "mdta/com.apple.quicktime.full-frame-rate-playback-intent"),
]

func pair(edited: URL, template: URL, output: URL, identifier: String) throws {
    if identifier.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { try fail("content identifier must not be blank") }
    try? FileManager.default.removeItem(at: output)
    let editedAsset = AVURLAsset(url: edited)
    let templateAsset = AVURLAsset(url: template)
    guard let editedVideo = editedAsset.tracks(withMediaType: .video).first else { try fail("no video track in \(edited.path)") }
    let markerTracks = templateAsset.tracks(withMediaType: .metadata)
    if markerTracks.isEmpty { try fail("\(template.path) has no timed metadata track; it is not a Live Photo movie") }
    let composition = AVMutableComposition()
    guard let video = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
        try fail("cannot create a composition video track")
    }
    let duration = min(editedAsset.duration, templateAsset.duration)
    try video.insertTimeRange(CMTimeRange(start: .zero, duration: duration), of: editedVideo, at: .zero)
    video.preferredTransform = editedVideo.preferredTransform
    for track in markerTracks {
        if let copy = composition.addMutableTrack(withMediaType: .metadata, preferredTrackID: kCMPersistentTrackID_Invalid) {
            try copy.insertTimeRange(CMTimeRange(start: .zero, duration: duration), of: track, at: .zero)
        }
    }
    guard let session = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
        try fail("cannot create an export session")
    }
    var metadata = templateAsset.metadata.filter { item in item.identifier.map { keptFileMetadata.contains($0) } ?? false }
    let identifierItem = AVMutableMetadataItem()
    identifierItem.identifier = .quickTimeMetadataContentIdentifier
    identifierItem.value = identifier as NSString
    identifierItem.dataType = kCMMetadataBaseDataType_UTF8 as String
    metadata.append(identifierItem)
    session.metadata = metadata
    session.outputURL = output
    session.outputFileType = .mov
    let done = DispatchSemaphore(value: 0)
    session.exportAsynchronously { done.signal() }
    done.wait()
    if session.status != .completed {
        try fail("export failed: status=\(session.status.rawValue) \(session.error?.localizedDescription ?? "")")
    }
}

func stamp(input: URL, identifier: String, output: URL) throws {
    if identifier.isEmpty { try fail("content identifier must not be empty") }
    guard let source = CGImageSourceCreateWithURL(input as CFURL, nil), let type = CGImageSourceGetType(source) else {
        try fail("cannot open \(input.path)")
    }
    guard let destination = CGImageDestinationCreateWithURL(output as CFURL, type, 1, nil) else {
        try fail("cannot create \(output.path)")
    }
    var properties = (CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]) ?? [:]
    var maker = properties[kCGImagePropertyMakerAppleDictionary] as? [String: Any] ?? [:]
    maker["17"] = identifier
    properties[kCGImagePropertyMakerAppleDictionary] = maker
    CGImageDestinationAddImageFromSource(destination, source, 0, properties as CFDictionary)
    if !CGImageDestinationFinalize(destination) { try fail("cannot write \(output.path)") }
}

func inspect(movie: URL) throws {
    let asset = AVURLAsset(url: movie)
    var markers: [Double] = []
    for track in asset.tracks(withMediaType: .metadata) {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        guard reader.canAdd(output) else { try fail("cannot read metadata track \(track.trackID)") }
        reader.add(output)
        let adaptor = AVAssetReaderOutputMetadataAdaptor(assetReaderTrackOutput: output)
        guard reader.startReading() else { try fail("metadata reader failed: \(String(describing: reader.error))") }
        while let group = adaptor.nextTimedMetadataGroup() {
            for item in group.items where item.identifier?.rawValue == "mdta/com.apple.quicktime.still-image-time" {
                markers.append(CMTimeGetSeconds(group.timeRange.start))
            }
        }
    }
    let identifier = asset.metadata.first { $0.identifier == .quickTimeMetadataContentIdentifier }?.stringValue ?? ""
    let keys = asset.metadata.compactMap { $0.identifier?.rawValue }
    try printJSON(["content_identifier": identifier, "duration_seconds": CMTimeGetSeconds(asset.duration),
                   "still_image_times": markers, "metadata_keys": keys])
}

func inspectPhoto(photo: URL) throws {
    guard let source = CGImageSourceCreateWithURL(photo as CFURL, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] else {
        try fail("cannot open \(photo.path)")
    }
    let maker = properties[kCGImagePropertyMakerAppleDictionary as String] as? [String: Any] ?? [:]
    try printJSON(["content_identifier": maker["17"] as? String ?? "",
                   "width": properties[kCGImagePropertyPixelWidth as String] ?? 0,
                   "height": properties[kCGImagePropertyPixelHeight as String] ?? 0])
}

let args = Array(CommandLine.arguments.dropFirst())
do {
    switch (args.first, args.count) {
    case ("pair", 5): try pair(edited: URL(fileURLWithPath: args[1]), template: URL(fileURLWithPath: args[2]),
                               output: URL(fileURLWithPath: args[3]), identifier: args[4])
    case ("stamp", 4): try stamp(input: URL(fileURLWithPath: args[1]), identifier: args[2], output: URL(fileURLWithPath: args[3]))
    case ("inspect", 2): try inspect(movie: URL(fileURLWithPath: args[1]))
    case ("inspect-photo", 2): try inspectPhoto(photo: URL(fileURLWithPath: args[1]))
    default:
        FileHandle.standardError.write("usage: live-photo-pair pair|stamp|inspect|inspect-photo ...\n".data(using: .utf8)!)
        exit(2)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}

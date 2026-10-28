import CFNetwork
import Foundation

private final class NoRedirectDelegate: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

func makeLocalSession() -> URLSession {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.httpShouldSetCookies = false
    configuration.urlCredentialStorage = nil
    configuration.connectionProxyDictionary = [
        kCFNetworkProxiesHTTPEnable as String: 0,
        kCFNetworkProxiesHTTPSEnable as String: 0,
        kCFNetworkProxiesSOCKSEnable as String: 0,
        kCFNetworkProxiesProxyAutoConfigEnable as String: 0,
        kCFNetworkProxiesProxyAutoDiscoveryEnable as String: 0
    ]
    return URLSession(configuration: configuration, delegate: NoRedirectDelegate(), delegateQueue: nil)
}
